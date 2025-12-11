import functools
import PIL
import numpy as np
from matplotlib import pyplot as plt
from enum import IntEnum
import gymnasium
from gymnasium.spaces import Box, MultiDiscrete
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
import pygame
from typing import Tuple

class Direction(IntEnum):
    """方向枚举，表示六边形网格的6个方向"""
    Right = 0      # 右
    UpRight = 1    # 右上
    UpLeft = 2     # 左上
    Left = 3       # 左
    DownLeft = 4   # 左下
    DownRight = 5  # 右下

class Position:
    """
    位置类，使用轴向坐标(q, r)表示
    满足约束: q + r + s = 0，其中s = -q - r
    """
    # 方向映射到坐标增量
    direction_map = {
        Direction.Right    : (+1,  0, -1),  # 右：q增加，s减少
        Direction.UpRight  : (+1, -1,  0),  # 右上：q增加，r减少
        Direction.UpLeft   : ( 0, -1, +1),  # 左上：r减少，s增加
        Direction.Left     : (-1,  0, +1),  # 左：q减少，s增加
        Direction.DownLeft : (-1, +1,  0),  # 左下：q减少，r增加
        Direction.DownRight: ( 0, +1, -1)   # 右下：r增加，s减少
    }

    def __init__(self, q: int, r: int):
        self.q = q
        self.r = r
        self.s = -q - r  # 计算s坐标

    def neighbor(self, direction: Direction, multiplier: int = 1):
        """获取指定方向上的相邻位置"""
        q_delta, r_delta, _ = Position.direction_map[direction]
        return Position(
            self.q + q_delta * multiplier,
            self.r + r_delta * multiplier
        )
    
    def __eq__(self, other):
        """位置相等性比较"""
        return self.q == other.q and self.r == other.r

class Move:
    """移动类，表示一个棋子移动"""
    END_TURN = "END_TURN"  # 结束回合的特殊标记
    
    def __init__(self, q: int, r: int, direction: Direction, is_jump: bool):
        """
        Args:
            q, r: 起始位置的轴向坐标
            direction: 移动方向
            is_jump: 是否为跳跃移动（跳过其他棋子）
        """
        self.position = Position(q, r)
        self.direction = direction
        self.is_jump = is_jump
        
    def moved_position(self):
        """计算移动后的位置"""
        multiplier = 2 if self.is_jump else 1  # 跳跃移动2格，普通移动1格
        return self.position.neighbor(self.direction, multiplier)
    
    @staticmethod
    def rotate60(move, times):
        """将移动旋转60度×times次"""
        assert move is not None
        q, r = move.position.q, move.position.r
        s = -q - r
        # 旋转坐标
        absolute_q, absolute_r, absolute_s = ChineseCheckers._rotate_60(q, r, s, -times)
        # 旋转方向
        absolute_direction = (move.direction - times) % 6
        return Move(absolute_q, absolute_r, absolute_direction, move.is_jump)
    
    @staticmethod
    def to_absolute_move(move, player):
        """将相对移动转换为绝对移动"""
        return Move.rotate60(move, -player) if move != Move.END_TURN else Move.END_TURN
    
    @staticmethod
    def to_relative_move(move, player):
        """将绝对移动转换为相对移动"""
        return Move.rotate60(move, player) if move != Move.END_TURN else Move.END_TURN

    def __str__(self):
        if (self == Move.END_TURN):
            return "Move.END_TURN"
        return f"Move({self.position.q}, {self.position.r}, {self.direction}, {self.is_jump})"
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other):
        """移动相等性比较"""
        if self is other:
            return True
        if isinstance(other, Move):
            return self.position == other.position and \
                   self.direction == other.direction and \
                   self.is_jump == other.is_jump
        else:
            return False

class ChineseCheckers:
    """
    跳棋游戏核心逻辑类
    使用立方体坐标(q, r, s)表示棋盘，满足 q + r + s = 0
    """
    
    # 颜色映射：-1为空，-2为无效，0-5为玩家颜色
    colors = {
        -1: (154, 132, 73),  # 空单元格
        0: (255, 0, 0),      # 红色
        # 1: (0, 0, 0),        # 黑色
        # 2: (255, 255, 0),    # 黄色
        3: (0, 255, 0),      # 绿色
        # 4: (0, 0, 255),      # 蓝色
        # 5: (255, 255, 255),  # 白色
    }

    OUT_OF_BOUNDS = -2  # 边界外
    EMPTY_SPACE = -1    # 空位置

    def __init__(self, triangle_size: int, render_mode: str = "rgb_array"):
        """
        初始化跳棋游戏
        
        Args:
            triangle_size: 三角区域大小（n）
            render_mode: 渲染模式
        """
        self.render_mode = render_mode
        self.n = triangle_size
        self.rotation = 0  # 当前视角旋转
        self.clock = None

        self.init_game()

        # 渲染相关
        self.window_size = 512  # PyGame窗口大小
        self.window = None

    def _set_rotation(self, player: int):
        """
        设置棋盘旋转，使指定玩家在棋盘顶部
        """
        self.rotation = player
        # 转换跳跃记录为当前视角
        self._jumps = list(map(lambda j: Move.to_relative_move(j, player), self._jumps))

    def _unset_rotation(self):
        """恢复棋盘旋转到绝对坐标"""
        assert self.rotation is not None
        self._jumps = list(map(lambda j: Move.to_relative_move(j, -self.rotation), self._jumps))
        self.rotation = None

    @staticmethod
    def _rotate_60(q: int, r: int, s: int, times: int):
        """
        绕原点顺时针旋转60度times次
        
        立方体坐标旋转公式：顺时针旋转60度相当于
        (q, r, s) -> (-r, -s, -q)
        """
        for _ in range(times % 6):
            q, r, s = -r, -s, -q
        return q, r, s

    @staticmethod
    def _rotate_60_about_pt(q: int, r: int, s: int, o_q: int, o_r: int, o_s: int, times: int):
        """
        绕指定点旋转60度
        先平移到原点，旋转，再平移回原位置
        """
        n_q = q - o_q
        n_r = r - o_r
        n_s = s - o_s
        n_q, n_r, n_s = ChineseCheckers._rotate_60(n_q, n_r, n_s, times)
        return n_q + o_q, n_r + o_r, n_s + o_s

    def init_game(self):
        """
        初始化游戏棋盘
        
        棋盘是3D数组，形状为(4n+1, 4n+1, 4n+1)
        值范围：-2(无效), -1(空), 0-5(玩家)
        """
        self._jumps = []  # 当前回合的跳跃记录
        self._legal_moves = None  # 合法移动缓存
        self._game_over = False   # 游戏结束标志

        def _fill_center_empty():
            """填充中心区域为空位置"""
            self._set_rotation(0)
            for q in range(-self.n, self.n + 1):
                for r in range(-self.n, self.n + 1):
                    s = -q - r
                    # 六边形区域条件
                    if abs(q) + abs(r) + abs(s) <= 2 * self.n:
                        self._set_coordinate(q, r, s, ChineseCheckers.EMPTY_SPACE)
            self._unset_rotation()

        def _fill_home_triangle(player: int):
            """填充玩家的起始三角区域"""
            self._set_rotation(player)
            for q, r, s in self._get_home_coordinates():
                self._set_coordinate(q, r, s, player)
            self._unset_rotation()
        
        # 初始化棋盘为无效位置
        self.board = ChineseCheckers.OUT_OF_BOUNDS * np.ones(
            (4 * self.n + 1, 4 * self.n + 1, 4 * self.n + 1), 
            dtype=np.int8
        )

        # 填充2个玩家的起始区域
        for player in [0, 3]:
            _fill_home_triangle(player)

        # 填充中心区域
        _fill_center_empty()
            
    def find_legal_moves(self, player: int):
        """
        查找指定玩家的所有合法移动并缓存
        
        遍历所有可能的位置、方向、移动类型
        """
        self._set_rotation(player)
        moves = []
        for q in range(-2 * self.n, 2 * self.n + 1):
            for r in range(-2 * self.n, 2 * self.n + 1):
                for direction in Direction:
                    for is_jump in [False, True]:
                        move = Move(q, r, direction, is_jump)
                        if self._is_single_move_legal(move):
                            moves.append(move)
        
        # 如果没有合法移动或可以结束回合，添加END_TURN
        if (len(moves) == 0 and not self._game_over) or self._is_single_move_legal(Move.END_TURN):
            moves.append(Move.END_TURN)
            
        self._legal_moves = moves
        self._unset_rotation()
    
    def get_legal_moves(self, player: int):
        """获取指定玩家的合法移动列表"""
        if self._legal_moves is None:
            self.find_legal_moves(player)
        return self._legal_moves

    def _get_home_coordinates(self):
        """
        生成当前玩家起始三角区域的坐标
        返回的是相对坐标
        """
        assert self.rotation is not None
        offset = np.array([1, -self.n - 1, self.n])  # 起始偏移
        # 三角区域坐标生成
        for i in range(self.n):
            for j in range(i, self.n):
                q, r, s = j, -i, i - j
                yield offset + np.array([q, r, s])
    
    def _home_values(self):
        """生成起始区域的所有值"""
        assert self.rotation is not None
        for q, r, s in self._get_home_coordinates():
            yield self._get_board_value(q, r, s)

    def _get_target_coordinates(self):
        """
        生成当前玩家目标三角区域的坐标
        目标区域与起始区域相对
        """
        assert self.rotation is not None
        offset = np.array([-self.n, self.n + 1, -1])  # 目标区域偏移
        for i in range(self.n):
            for j in range(0, self.n - i):
                q, r, s = j, i, -i - j
                yield offset + np.array([q, r, s])

    def get_target_coordinates(self, player):
        """获取指定玩家的目标区域坐标"""
        self._set_rotation(player)
        yield from self._get_target_coordinates()
        self._unset_rotation()

    def _target_values(self):
        """生成目标区域的所有值"""
        assert self.rotation is not None
        for q, r, s in self._get_target_coordinates():
            yield self._get_board_value(q, r, s)
    
    def _in_bounds(self, q: int, r: int, s: int):
        """检查坐标是否在有效棋盘范围内"""
        board_q, board_r, board_s = q + 2 * self.n, r + 2 * self.n, s + 2 * self.n
        if (board_q < 0 or board_q >= 4 * self.n + 1 or \
            board_r < 0 or board_r >= 4 * self.n + 1 or \
            board_s < 0 or board_s >= 4 * self.n + 1):
            return False
        
        start = self._get_board_value(q, r, s)
        return start != ChineseCheckers.OUT_OF_BOUNDS
        
    def _get_board_value(self, q: int, r: int, s: int):
        """
        获取指定坐标的棋盘值
        考虑当前旋转角度
        """
        assert self.rotation is not None
        # 旋转到绝对坐标
        rotated_q, rotated_r, rotated_s = self._rotate_60(q, r, s, self.rotation)
        # 转换为数组索引
        board_q, board_r, board_s = rotated_q + 2 * self.n, rotated_r + 2 * self.n, rotated_s + 2 * self.n
        return self.board[board_q, board_r, board_s]

    def _set_coordinate(self, q: int, r: int, s: int, value: int):
        """设置指定坐标的棋盘值"""
        assert self.rotation is not None
        rotated_q, rotated_r, rotated_s = self._rotate_60(q, r, s, self.rotation)
        board_q, board_r, board_s = rotated_q + 2 * self.n, rotated_r + 2 * self.n, rotated_s + 2 * self.n
        self.board[board_q, board_r, board_s] = value
    
    @staticmethod
    def _cube_to_axial(q: int, r: int, s: int):
        """立方体坐标转轴向坐标（忽略s）"""
        return q, r
    
    @staticmethod
    def _axial_to_cube(q: int, r: int):
        """轴向坐标转立方体坐标（计算s）"""
        return q, r, -q - r
    
    def _get_axial_board(self):
        """获取当前视角下的轴向棋盘"""
        assert self.rotation is not None 
        result = -2 * np.ones((4 * self.n + 1, 4 * self.n + 1), dtype=np.int8)
        for q in range(-2 * self.n, 2 * self.n + 1):
            for r in range(-2 * self.n, 2 * self.n + 1):
                s = -q - r
                # 检查是否在六边形区域内
                if abs(q) + abs(r) + abs(s) <= 4 * self.n + 1:
                    result[q, r] = self._get_board_value(q, r, s)
        # 旋转玩家编号
        result = np.vectorize(self._rotate_player_number)(result)
        return result   

    def _is_player(self, value):
        """检查是否为玩家编号"""
        return 0 <= value < 6

    def _rotate_player_number(self, peg):
        """旋转玩家编号（从绝对坐标到相对坐标）"""
        assert self.rotation is not None
        return (peg - self.rotation) % 6 if self._is_player(peg) else peg
    
    def get_axial_board(self, player):
        """获取指定玩家视角的轴向棋盘"""
        self._set_rotation(player)
        board = self._get_axial_board()
        self._unset_rotation()
        return board
    
    def _is_single_move_legal(self, move: Move):
        """
        检查单个移动是否合法
        假设当前玩家的起始区域在棋盘顶部
        """
        assert self.rotation is not None

        if self._game_over:
            return False

        player = self.rotation

        # 检查结束回合的条件
        if (move == Move.END_TURN):
            return len(self._jumps) > 0  # 只有在有跳跃时才能结束回合

        # 检查起始位置是否在边界内
        if (not self._in_bounds(move.position.q, move.position.r, move.position.s)):
            return False
        
        # 起始位置必须是当前玩家的棋子
        start_position_value = self._get_board_value(
            move.position.q, 
            move.position.r, 
            move.position.s
        )
        if (start_position_value != player):
            return False
        
        # 目标位置必须在边界内
        moved_to = move.moved_position()
        if (not self._in_bounds(moved_to.q, moved_to.r, moved_to.s)):
            return False
        
        # 目标位置必须是空的
        moved_to_value = self._get_board_value(moved_to.q, moved_to.r, moved_to.s)
        if (moved_to_value != ChineseCheckers.EMPTY_SPACE):
            return False
        
        # 跳跃移动的特殊检查
        if move.is_jump:
            # 检查中间是否有棋子可以跳过
            direct_neighbor = move.position.neighbor(move.direction)
            if (not self._in_bounds(direct_neighbor.q, direct_neighbor.r, direct_neighbor.s)):
                return False
            direct_neighbor_value = self._get_board_value(
                direct_neighbor.q, 
                direct_neighbor.r, 
                direct_neighbor.s
            )
            if (direct_neighbor_value == ChineseCheckers.EMPTY_SPACE):
                return False
            
            # 多段跳跃规则检查
            if (len(self._jumps) > 0):
                # 只能移动同一个棋子
                last_jump = self._jumps[-1]
                prev_jumped_position = last_jump.position.neighbor(last_jump.direction, 2)
                if (prev_jumped_position != move.position):
                    return False
                
                # 防止循环跳跃
                jumped_position = move.position.neighbor(move.direction, 2)
                for prev_jump in self._jumps:
                    if (jumped_position == prev_jump.position):
                        return False
        else:
            # 非跳跃移动时，不能有之前的跳跃
            if len(self._jumps) != 0:
                return False

        return True
    
    def is_move_legal(self, move: Move, player: int) -> bool:
        """检查移动对指定玩家是否合法"""
        if self._legal_moves is None:
            self.find_legal_moves(player)
        self._set_rotation(player)
        is_legal = move in self._legal_moves
        self._unset_rotation()
        return is_legal
    
    def move(self, player: int, move: Move) -> int:
        """
        执行移动
        
        Args:
            player: 执行移动的玩家
            move: 移动对象
            
        Returns:
            int: 移动后的棋子数（实际未使用，返回move本身）
        """
        if (not self.is_move_legal(move, player)):
            print(f"{move} is not legal for player {player}")
            return None
        
        if move == Move.END_TURN:
            self._jumps.clear()     # 清空跳跃记录
            self._legal_moves = None  # 重置合法移动缓存
            return move
        
        self._set_rotation(player)
        
        # 执行移动
        src_pos = move.position
        self._set_coordinate(src_pos.q, src_pos.r, src_pos.s, ChineseCheckers.EMPTY_SPACE)  # 清空起始位置
        dst_pos = move.moved_position()
        self._set_coordinate(dst_pos.q, dst_pos.r, dst_pos.s, player)  # 放置棋子到目标位置
        
        # 更新跳跃记录
        if move.is_jump:
            self._jumps.append(move)
        else:
            self._jumps.clear()  # 非跳跃移动清除跳跃记录
            
        self._legal_moves = None
        
        # 检查是否获胜
        if self._did_player_win():
            self._game_over = True
            
        self._unset_rotation()
        return move

    def get_jumps(self, player: int):
        """获取指定玩家的跳跃记录"""
        return [Move.to_relative_move(jump, player) for jump in self._jumps]

    def get_last_jump(self, player: int):
        """获取指定玩家的最后一次跳跃"""
        if len(self._jumps) > 0:
            return Move.to_relative_move(self._jumps[-1], player)
        return None

    def _did_player_win(self) -> bool:
        """检查当前玩家是否获胜（所有目标位置都是自己的棋子）"""
        return all([value == self.rotation for value in self._target_values()])

    def did_player_win(self, player: int) -> bool:
        """检查指定玩家是否获胜"""
        self._set_rotation(player)
        did_win = self._did_player_win()
        self._unset_rotation()
        return did_win
    
    def is_game_over(self) -> bool:
        """检查游戏是否结束"""
        return self._game_over
    
    def render(self, player: int = 0):
        """渲染游戏画面"""
        self._set_rotation(player)
        frame = None
        frame = self._render_frame()
        self._unset_rotation()
        return frame

    def _render_frame(self):
        """渲染一帧游戏画面"""
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # 创建画布
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((241, 212, 133))  # 填充背景色

        def axial_to_pixel(q: int, r: int):
            """轴向坐标转换为像素坐标"""
            l = 20  # 六边形大小
            screen_center_x, screen_center_y = self.window_size / 2, self.window_size / 2
            # 六边形网格坐标转换公式
            return screen_center_x + l * np.sqrt(3) * (q + 0.5 * r), \
                screen_center_y + l * 1.5 * r

        # 获取轴向棋盘
        axial_board = self._get_axial_board()
        
        # 绘制所有棋子
        for q in range(-2 * self.n, 2 * self.n + 1):
            for r in range(-2 * self.n, 2 * self.n + 1):
                pixel_x, pixel_y = axial_to_pixel(q, r)
                cell = axial_board[q, r]
                if cell == ChineseCheckers.OUT_OF_BOUNDS:
                    continue  # 跳过无效位置
                else:
                    # 绘制棋子（圆形）
                    pygame.draw.circle(
                        canvas,
                        self.colors[cell],
                        (pixel_x, pixel_y),
                        8,  # 棋子半径
                    )
                    
        if self.render_mode == "human":
            # 将画布内容显示到窗口
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # 控制帧率
            self.clock.tick(12)
        else:  # rgb_array模式
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        