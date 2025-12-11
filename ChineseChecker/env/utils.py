from .game import ChineseCheckers, Move, Position
import numpy as np


def action_to_move(action: int, n: int):
    """
    将动作索引转换为Move对象
    
    Args:
        action: 动作索引（0到action_space_dim-1）
        n: 三角区域大小
        
    Returns:
        Move: 对应的移动对象或END_TURN
        
    动作编码：平坦化的 (q, r, direction, is_jump) 四元组
    维度: (4n+1)^2 × 6 × 2 + 1
    """
    if (action == (4 * n + 1) ** 2 * 6 * 2):
        return Move.END_TURN  # 最后一个动作是结束回合
    
    index = action
    index, is_jump = divmod(index, 2)     # 提取是否跳跃
    index, direction = divmod(index, 6)   # 提取方向
    _q, _r = divmod(index, 4 * n + 1)     # 提取坐标索引
    q, r = _q - 2 * n, _r - 2 * n         # 转换为相对坐标
    return Move(q, r, direction, bool(is_jump))

def move_to_action(move: Move, n: int):
    """
    将Move对象转换为动作索引
    
    Args:
        move: 移动对象
        n: 三角区域大小
        
    Returns:
        int: 动作索引
    """
    if (move == Move.END_TURN):
        return (4 * n + 1) ** 2 * 6 * 2  # 结束回合的特殊索引
    
    # 提取移动参数
    q, r, direction, is_jump = move.position.q, move.position.r, move.direction, move.is_jump
    # 编码公式: is_jump + 2*(direction + 6*((r+2n) + (4n+1)*(q+2n)))
    index = int(is_jump) + 2 * (direction + 6 * ((r + 2 * n) + (4 * n + 1) * (q + 2 * n)))
    return index

def get_legal_move_mask(board: ChineseCheckers, player: int):
    """
    获取合法动作掩码
    
    Args:
        board: 游戏棋盘
        player: 玩家编号
        
    Returns:
        np.array: 形状为(action_space_dim,)的掩码数组，合法动作为1，否则为0
    """
    # 初始化掩码（全0）
    mask = np.zeros((4 * board.n + 1, 4 * board.n + 1, 6, 2), dtype=np.int8).flatten()
    # 添加结束回合位
    mask = np.append(mask, np.int8(0))
    
    # 为每个合法动作设置掩码为1
    for move in board.get_legal_moves(player):
        mask[move_to_action(move, board.n)] = np.int8(1)
        
    return mask

def rotate_observation(observation: np.array, player: int):
    """
    旋转观察中的玩家通道
    
    Args:
        observation: 观察数组
        player: 当前玩家编号
        
    Returns:
        np.array: 旋转后的观察
    """
    return np.roll(observation, player)  # 循环移位玩家通道