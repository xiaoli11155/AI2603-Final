import csv
import datetime
import os
import copy
import numpy as np
import glob
import argparse
from pathlib import Path
from tqdm import tqdm

from gymnasium.spaces import Box, Discrete

from pettingzoo.classic import rps_v2

import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.algorithms.ppo import (
    PPO,
    PPOConfig,
)
from ray.rllib.models import ModelCatalog
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.tune.logger import pretty_print
from ray.rllib.policy.policy import Policy

from ChineseChecker.env.game import Direction, Move, Position

# Random Policy 
class ChineseCheckersRandomPolicy(Policy):
    def __init__(self, triangle_size=4, config={}):
        observation_space = Box(low=0, high=1, shape=((4 * triangle_size + 1) * (4 * triangle_size + 1) * 4,), dtype=np.int8)
        action_space = Discrete((4 * triangle_size + 1) * (4 * triangle_size + 1) * 6 * 2 + 1)
        super().__init__(observation_space, action_space, config)
        self.action_space = action_space

    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None, prev_reward_batch=None, info_batch=None, episodes=None, **kwargs):
        actions = []
        for obs in obs_batch:
            action = self.action_space.sample(obs["action_mask"])
            actions.append(action)
        return actions, [], {}

    def compute_single_action(self, obs, state=None, prev_action=None, prev_reward=None, info=None, episode=None, **kwargs):
        return self.compute_actions([obs], state_batches=[state], prev_action_batch=[prev_action], prev_reward_batch=[prev_reward], info_batch=[info], episodes=[episode], **kwargs)[0]

# TODO: Greedy Policy
class GreedyPolicy(Policy):
    def __init__(self, triangle_size=4, config={}):
        # 观察空间：扁平化的棋盘状态 + 动作掩码
        observation_space = Box(low=0, high=1, shape=((4 * triangle_size + 1) * (4 * triangle_size + 1) * 4,), dtype=np.int8)
        # 动作空间：所有可能的移动 + 结束回合
        action_space = Discrete((4 * triangle_size + 1) * (4 * triangle_size + 1) * 6 * 2 + 1)
        super().__init__(observation_space, action_space, config)
        self.triangle_size = triangle_size
        self.action_space_dim = action_space.n
        
    def _action_to_move(self, action: int):
        """将动作索引转换为Move对象（从utils.py复制过来的逻辑）"""
        n = self.triangle_size
        
        if action == (4 * n + 1) ** 2 * 6 * 2:
            return Move.END_TURN
        
        index = action
        index, is_jump = divmod(index, 2)     # 提取是否跳跃
        index, direction = divmod(index, 6)   # 提取方向
        _q, _r = divmod(index, 4 * n + 1)     # 提取坐标索引
        q, r = _q - 2 * n, _r - 2 * n         # 转换为相对坐标
        return Move(q, r, direction, bool(is_jump))
    
    def _move_to_action(self, move: Move):
        """将Move对象转换为动作索引（从utils.py复制过来的逻辑）"""
        n = self.triangle_size
        
        if move == Move.END_TURN:
            return (4 * n + 1) ** 2 * 6 * 2
        
        q, r, direction, is_jump = move.position.q, move.position.r, move.direction, move.is_jump
        index = int(is_jump) + 2 * (direction + 6 * ((r + 2 * n) + (4 * n + 1) * (q + 2 * n)))
        return index
    
    def _calculate_move_score(self, move, player, board_observation):
        """
        计算移动的得分（奖励估计）
        基于环境中的奖励规则
        """
        n = self.triangle_size
        
        if move == Move.END_TURN:
            # 结束回合的得分较低，除非没有其他合法移动
            return 0.0
        
        # 基础得分
        score = 0.0
        # 1. 鼓励向目标区域移动，惩罚远离目标区域
        if move.direction in [Direction.DownLeft, Direction.DownRight]:
            move_distance = 2 if move.is_jump else 1
            score += 1.0 * move_distance
        elif move.direction in [Direction.UpLeft, Direction.UpRight]:
            move_distance = 2 if move.is_jump else 1
            score -= 1.0 * move_distance

        return score
    
    def _get_player_from_observation(self, observation):
        """
        从观察中推断当前玩家
        观察包含4个通道：当前玩家棋子、其他玩家棋子、跳跃起始位置、上次跳跃目标位置
        通过查找哪个通道有棋子来推断
        """
        n = self.triangle_size
        board_size = 4 * n + 1
        
        # 重塑观察为通道形式
        channels = observation["observation"].reshape(board_size, board_size, 4)
        
        # 第一个通道应该是当前玩家的棋子
        # 但为了安全，我们检查哪个通道有最多的棋子
        player0_pieces = np.sum(channels[:, :, 0])  # 通道0：当前玩家棋子
        player3_pieces = np.sum(channels[:, :, 1])  # 通道1：其他玩家棋子
        
        # 如果有棋子，返回对应玩家
        if player0_pieces > 0:
            return 0
        elif player3_pieces > 0:
            return 3
        else:
            # 默认返回0
            return 0
    
    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None, 
                       prev_reward_batch=None, info_batch=None, episodes=None, **kwargs):
        """
        计算一批观察的动作
        使用贪心策略：选择得分最高的合法动作
        """
        actions = []
        
        for i, obs in enumerate(obs_batch):
            # 获取动作掩码
            action_mask = obs["action_mask"]
            
            # 推断当前玩家
            player = self._get_player_from_observation(obs)
            
            # 初始化最佳动作和最高得分
            best_action = None
            best_score = -float('inf')
            
            # 遍历所有可能的动作
            for action_idx in range(self.action_space_dim):
                # 检查动作是否合法
                if action_mask[action_idx] == 1:
                    # 转换为Move对象
                    move = self._action_to_move(action_idx)
                    
                    # 计算动作得分
                    score = self._calculate_move_score(move, player, obs)
                    
                    # 更新最佳动作
                    if score > best_score:
                        best_score = score
                        best_action = action_idx
            
            # 如果没有找到合法动作（理论上不会发生），选择第一个合法动作
            if best_action is None:
                # 查找第一个合法动作
                for action_idx in range(self.action_space_dim):
                    if action_mask[action_idx] == 1:
                        best_action = action_idx
                        break
            
            # 如果没有合法动作，选择结束回合
            if best_action is None:
                best_action = self.action_space_dim - 1  # END_TURN动作
            
            actions.append(best_action)
        
        return actions, [], {}
    
    def compute_single_action(self, obs, state=None, prev_action=None, 
                              prev_reward=None, info=None, episode=None, **kwargs):
        """
        计算单个观察的动作
        """
        return self.compute_actions(
            [obs], 
            state_batches=[state], 
            prev_action_batch=[prev_action], 
            prev_reward_batch=[prev_reward], 
            info_batch=[info], 
            episodes=[episode], 
            **kwargs
        )[0]

# TODO: Your Policy
from ChineseChecker.env.game import (
    ChineseCheckers,
    Direction,
    Move,
    Position
)


class MinimaxPolicy(Policy):
    """
    Alpha-Beta Minimax Policy for Chinese Checkers
    (RLlib-compatible, Greedy-style interface)
    """

    def __init__(self, triangle_size=4, depth=2, config=None):
        self.triangle_size = triangle_size
        self.depth = depth

        observation_space = Box(
            low=0,
            high=1,
            shape=((4 * triangle_size + 1) ** 2 * 4,),
            dtype=np.int8
        )
        action_space = Discrete((4 * triangle_size + 1) ** 2 * 6 * 2 + 1)

        super().__init__(observation_space, action_space, config or {})
        self.action_space_dim = action_space.n

    def _move_to_action(self, move: Move):
        n = self.triangle_size
        if move == Move.END_TURN:
            return (4 * n + 1) ** 2 * 6 * 2

        q, r = move.position.q, move.position.r
        return int(move.is_jump) + 2 * (
            move.direction + 6 * ((r + 2 * n) + (4 * n + 1) * (q + 2 * n))
        )


    def _reconstruct_game(self, obs):
        n = self.triangle_size
        dim = 4 * n + 1
    
        game = ChineseCheckers(n)
        game._jumps = []
        game.current_player = 0
        game.init_game()
        game.rotation = 0
        game.board[game.board >= 0] = ChineseCheckers.EMPTY_SPACE
        #print(obs["observation"].shape)
        channels = obs["observation"].reshape(dim, dim, 4)

        jump_sources = []
        last_jump_dest = None
        #print(1234)
        for q in range(-2 * n, 2 * n + 1):
            for r in range(-2 * n, 2 * n + 1):
                s = -q - r

                if abs(s) > 2 * n:
                    continue
                i, j = q, r 

                if channels[i, j, 0] == 1:
                    #print(q,r,s,0)
                    game._set_coordinate(q, r, s, 0)

                if channels[i, j, 1] == 1:
                    #print(q,r,s,1)
                    game._set_coordinate(q, r, s, 3)

                if channels[i, j, 2] == 1:
                    #print(q,r,s,2)
                    jump_sources.append(Position(q, r))

                if channels[i, j, 3] == 1:
                    #print(q,r,s,3)
                    last_jump_dest = Position(q, r)
                #else:
                    #print(q,r,s,-1)


        if last_jump_dest:
            prev = None
            for src in jump_sources:
                for d in Direction:
                    if src.neighbor(d, 2) == last_jump_dest:
                        prev = src
                        game._jumps.append(Move(src.q, src.r, d, True))
                        break
                if prev:
                    break

            for src in jump_sources:
                if src != prev and src != last_jump_dest:
                    game._jumps.insert(
                    0, Move(src.q, src.r, Direction.Right, True)
                )

        return game
            
    def _evaluate_state(self, game):
        idx = np.where((game.board == 0) | (game.board == 3))
        rs = idx[1]**2
        return np.sum(rs)
    
    def _minimax(self, game, depth, alpha, beta, maximizing):
        if depth == 0 or game.is_game_over():
            return self._evaluate_state(game)

        player = 0 if maximizing else 3
        moves = game.get_legal_moves(player)

        if maximizing:
            value = -float("inf")
            for move in moves:
                g = copy.deepcopy(game)
                g.move(player, move)
                next_max = maximizing
                next_depth = depth
                if move == Move.END_TURN or not move.is_jump:
                    next_max = not maximizing
                    next_depth -= 1

                value = max(
                    value,
                    self._minimax(g, next_depth, alpha, beta, next_max)
                )
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = float("inf")
            for move in moves:
                g = copy.deepcopy(game)
                g.move(player, move)
                next_max = maximizing
                next_depth = depth
                if move == Move.END_TURN or not move.is_jump:
                    next_max = not maximizing
                    next_depth -= 1

                value = min(
                    value,
                    self._minimax(g, next_depth, alpha, beta, next_max)
                )
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

   
    def compute_actions(
        self,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        **kwargs
    ):
        actions = []

        for obs in obs_batch:
            game = self._reconstruct_game(obs)
            best_value = -float("inf")
            best_move = None
            alpha, beta = -float("inf"), float("inf")
            for move in game.get_legal_moves(0):

                g = copy.deepcopy(game)
                g.move(0, move)

                if move == Move.END_TURN or not move.is_jump:
                    val = self._minimax(g, self.depth - 1, alpha, beta, False)
                else:
                    val = self._minimax(g, self.depth, alpha, beta, True)

                if val > best_value:
                    best_value = val
                    best_move = move

                alpha = max(alpha, val)

            if best_move is None:
                actions.append(self.action_space_dim - 1)
            else:
                #print(best_move)
                actions.append(self._move_to_action(best_move))

        return actions, [], {}

    def compute_single_action(
        self,
        obs,
        state=None,
        prev_action=None,
        prev_reward=None,
        info=None,
        episode=None,
        **kwargs
    ):
        return self.compute_actions(
            [obs],
            state_batches=[state],
            prev_action_batch=[prev_action],
            prev_reward_batch=[prev_reward],
            info_batch=[info],
            episodes=[episode],
            **kwargs
        )[0]



  
if __name__ == "__main__":
    pass