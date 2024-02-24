# 有効な手かどうかをチェック
def is_valid_move(board, row, col, player):
    # すでに駒が置かれている場合は無効な移動
    if board[row][col] != 0:
        return False

    # 8方向の移動をチェック
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]
    valid_move = False

    for dr, dc in directions:
        r, c = row + dr, col + dc
        found_opponent = False

        while 0 <= r < 8 and 0 <= c < 8:
            if board[r][c] == 0:
                break
            elif found_opponent == False and board[r][c] == player:
                break
            elif board[r][c] != player:
                found_opponent = True
            elif board[r][c] == player and found_opponent:
                valid_move = True
                break
            r, c = r + dr, c + dc

    return valid_move

# 石を置く処理
def make_move(board, row, col, player):
    directions = [
        (0, 1), (0, -1), (1, 0), (-1, 0),
        (1, 1), (-1, -1), (-1, 1), (1, -1)
    ]

    board[row][col] = player

    for dr, dc in directions:
        r, c = row + dr, col + dc
        if 0 <= r < 8 and 0 <= c < 8 and board[r][c] != player and board[r][c] != 0:
            while 0 <= r < 8 and 0 <= c < 8 and board[r][c] != 0:
                r += dr
                c += dc
                if 0 <= r < 8 and 0 <= c < 8 and board[r][c] == player:
                    tr, tc = row + dr, col + dc
                    while (tr, tc) != (r, c):
                        board[tr][tc] = player
                        tr += dr
                        tc += dc
                    break

def pass_function(board, player):
    for row in range(8):
        for col in range(8):
            if is_valid_move(board, row, col, player):
                return True
    return False
