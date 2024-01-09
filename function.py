# 有効な手かどうかをチェック
def is_valid_move(board, row, col, player):
    if board[row][col] != 0:
        return False

    directions = [
        (0, 1), (0, -1), (1, 0), (-1, 0),
        (1, 1), (-1, -1), (-1, 1), (1, -1)
    ]

    for dr, dc in directions:
        r, c = row + dr, col + dc
        if 0 <= r < 8 and 0 <= c < 8 and board[r][c] != player and board[r][c] != 0:
            while 0 <= r < 8 and 0 <= c < 8 and board[r][c] != 0:
                r += dr
                c += dc
                if 0 <= r < 8 and 0 <= c < 8 and board[r][c] == player:
                    return True

    return False

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