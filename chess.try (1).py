import pygame
import sys
import chess
import chess.polyglot
import os
import random
import time
import threading
from collections import defaultdict

# Инициализация Pygame
pygame.init()
pygame.font.init()

# Размеры и цвета - ЯРКАЯ цветовая схема
SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 850
BOARD_SIZE = 640
SQUARE_SIZE = BOARD_SIZE // 8
MARGIN = 20

# Яркая палитра
COLORS = {
    'BACKGROUND': (15, 25, 40),
    'BOARD_LIGHT': (245, 222, 179),
    'BOARD_DARK': (205, 133, 63),
    'ACCENT': (0, 150, 255),
    'HIGHLIGHT': (255, 215, 0, 180),
    'LEGAL_MOVE': (50, 205, 50, 160),
    'LAST_MOVE': (255, 140, 0, 160),
    'PANEL_BG': (30, 40, 60),
    'TEXT': (255, 255, 255),
    'BUTTON': (0, 120, 215),
    'BUTTON_HOVER': (0, 180, 255),
    'PROGRESS': (0, 200, 100),
    'SUCCESS': (0, 200, 50),
    'ERROR': (255, 50, 50),
    'WARNING': (255, 200, 0)
}


# Шрифты
def get_font(size, bold=False):
    try:
        if bold:
            return pygame.font.SysFont('Arial', size, bold=True)
        return pygame.font.SysFont('Arial', size)
    except:
        return pygame.font.Font(None, size)


FONTS = {
    'TITLE': get_font(64, True),
    'HEADER': get_font(36, True),
    'BUTTON': get_font(28),
    'INFO': get_font(24),
    'SMALL': get_font(20),
    'PIECE': get_font(48)
}

# Шахматные символы
PIECE_SYMBOLS = {
    'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟',
    'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔', 'P': '♙'
}

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("♔ Шахматы Python AI ♚")
clock = pygame.time.Clock()

# Глобальные переменные
board = chess.Board()
selected_square = None
legal_moves = []
last_move = None
game_over = False
player_color = chess.WHITE
current_state = "MENU"
difficulty = 2
is_thinking = False
think_start_time = 0
progress_value = 0
status_message = "Готов к игре!"
status_color = COLORS['SUCCESS']
thinking_depth = 0
ai_move_history = []


class PurePythonAI:
    """Чисто Python шахматный ИИ без внешних зависимостей"""

    def __init__(self):
        self.initialized = True
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        self.opening_book = self.create_opening_book()
        self.move_cache = {}
        print("✅ Python Chess AI инициализирован")

    def create_opening_book(self):
        """Создаёт встроенную базу дебютов"""
        return {
            # Стандартные дебюты
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -": ["e2e4", "d2d4", "g1f3", "c2c4"],
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3": ["e7e5", "c7c5", "e7e6", "c7c6"],
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6": ["g1f3", "b1c3", "f1c4"],
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6": ["g1f3", "d2d4", "b1c3"],
        }

    def evaluate_position(self, board_state):
        """Оценка позиции"""
        score = 0

        # 1. Материальный баланс
        for square in chess.SQUARES:
            piece = board_state.piece_at(square)
            if piece:
                value = self.piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    score += value
                else:
                    score -= value

        # 2. Активность фигур (контроль центра)
        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        for move in board_state.legal_moves:
            if move.to_square in center_squares:
                if board_state.turn == chess.WHITE:
                    score += 10
                else:
                    score -= 10

        # 3. Безопасность короля
        if board_state.is_check():
            if board_state.turn == chess.WHITE:
                score -= 50
            else:
                score += 50

        # 4. Мобильность
        mobility = len(list(board_state.legal_moves))
        if board_state.turn == chess.WHITE:
            score += mobility * 2
        else:
            score -= mobility * 2

        # Возвращаем оценку с точки зрения белых
        return score if board_state.turn == chess.WHITE else -score

    def minimax(self, board_state, depth, alpha, beta, maximizing_player):
        """Алгоритм минимакс с альфа-бета отсечением"""
        if depth == 0 or board_state.is_game_over():
            return self.evaluate_position(board_state), None

        best_move = None
        legal_moves_list = list(board_state.legal_moves)

        if maximizing_player:
            max_eval = -float('inf')
            for move in legal_moves_list:
                board_state.push(move)
                eval_score, _ = self.minimax(board_state, depth - 1, alpha, beta, False)
                board_state.pop()

                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move

                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in legal_moves_list:
                board_state.push(move)
                eval_score, _ = self.minimax(board_state, depth - 1, alpha, beta, True)
                board_state.pop()

                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move

                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def get_best_move(self, board_state, difficulty_level):
        """Получение лучшего хода"""
        global thinking_depth

        # Сначала проверяем базу дебютов
        fen_key = board_state.fen().split(' ')[0]
        if fen_key in self.opening_book:
            for move_uci in self.opening_book[fen_key]:
                try:
                    from_square = chess.parse_square(move_uci[:2])
                    to_square = chess.parse_square(move_uci[2:4])
                    if len(move_uci) == 5:
                        promotion = {"q": chess.QUEEN, "r": chess.ROOK,
                                     "b": chess.BISHOP, "n": chess.KNIGHT}[move_uci[4]]
                        move = chess.Move(from_square, to_square, promotion=promotion)
                    else:
                        move = chess.Move(from_square, to_square)

                    if move in board_state.legal_moves:
                        print(f"📚 Ход из базы дебютов: {move_uci}")
                        return move
                except Exception as e:
                    continue

        # Устанавливаем глубину поиска по сложности
        if difficulty_level == 1:
            thinking_depth = 2
            depth = 2
        elif difficulty_level == 2:
            thinking_depth = 3
            depth = 3
        elif difficulty_level == 3:
            thinking_depth = 4
            depth = 4
        else:  # Эксперт
            thinking_depth = 5
            depth = 5

        # Используем минимакс
        try:
            _, best_move = self.minimax(board_state, depth, -float('inf'), float('inf'),
                                        board_state.turn == chess.WHITE)
        except Exception as e:
            print(f"Ошибка в минимаксе: {e}")
            best_move = None

        if best_move is None or best_move not in board_state.legal_moves:
            # Резерв: выбираем случайный легальный ход
            legal_moves_list = list(board_state.legal_moves)
            if legal_moves_list:
                # Предпочитаем шах или взятие
                for move in legal_moves_list:
                    if board_state.gives_check(move):
                        return move
                for move in legal_moves_list:
                    if board_state.is_capture(move):
                        return move
                return random.choice(legal_moves_list)

        return best_move


# Создаём ИИ
ai_engine = PurePythonAI()


class Button:
    """Красивые кнопки с анимацией"""

    def __init__(self, x, y, width, height, text, color=COLORS['BUTTON']):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = COLORS['BUTTON_HOVER']
        self.hovered = False
        self.animation = 0

    def draw(self, surface):
        # Анимация наведения
        if self.hovered and self.animation < 10:
            self.animation += 1
        elif not self.hovered and self.animation > 0:
            self.animation -= 1

        color = self.hover_color if self.hovered else self.color
        anim_offset = self.animation

        # Рисуем кнопку с тенью
        shadow_rect = pygame.Rect(self.rect.x + 3, self.rect.y + 3,
                                  self.rect.width, self.rect.height)
        pygame.draw.rect(surface, (0, 0, 0, 100), shadow_rect, border_radius=12)

        # Основная кнопка
        pygame.draw.rect(surface, color, self.rect, border_radius=12)
        pygame.draw.rect(surface, COLORS['TEXT'], self.rect, 3, border_radius=12)

        # Текст
        text_surf = FONTS['BUTTON'].render(self.text, True, COLORS['TEXT'])
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def check_hover(self, pos):
        self.hovered = self.rect.collidepoint(pos)
        return self.hovered

    def is_clicked(self, pos, event_type):
        return event_type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(pos)


class ProgressIndicator:
    """Анимированный индикатор прогресса"""

    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.value = 0
        self.pulse = 0
        self.pulse_dir = 1

    def update(self, thinking, think_time=0):
        if thinking:
            self.value = (self.value + 3) % 100
            self.pulse = (self.pulse + self.pulse_dir * 5) % 100
            if self.pulse >= 100:
                self.pulse_dir = -1
            elif self.pulse <= 0:
                self.pulse_dir = 1
        else:
            self.value = 0
            self.pulse = 0

    def draw(self, surface, thinking=False, depth=0):
        # Фон
        pygame.draw.rect(surface, (40, 50, 70), self.rect, border_radius=8)

        if thinking:
            # Анимированный прогресс-бар
            bar_width = int(self.rect.width * self.value / 100)
            bar_rect = pygame.Rect(self.rect.x, self.rect.y, bar_width, self.rect.height)

            # Пульсирующий цвет
            pulse_color = (
                COLORS['PROGRESS'][0] + int(self.pulse / 2),
                COLORS['PROGRESS'][1],
                COLORS['PROGRESS'][2]
            )
            pygame.draw.rect(surface, pulse_color, bar_rect, border_radius=8)

            # Текст
            status = FONTS['INFO'].render(f"AI анализирует (глубина {depth})...",
                                          True, COLORS['TEXT'])
            surface.blit(status, (self.rect.x, self.rect.y - 35))

        # Обводка
        pygame.draw.rect(surface, COLORS['ACCENT'], self.rect, 2, border_radius=8)


# Создаём UI с чёткими отступами
def create_menu_buttons():
    button_width, button_height = 320, 70
    start_x = (SCREEN_WIDTH - button_width) // 2
    return [
        Button(start_x, 320, button_width, button_height, "♔ ИГРАТЬ БЕЛЫМИ"),
        Button(start_x, 405, button_width, button_height, "♚ ИГРАТЬ ЧЁРНЫМИ"),
        Button(start_x, 490, button_width, button_height, "⚙ НАСТРОЙКИ СЛОЖНОСТИ"),
        Button(start_x, 575, button_width, button_height, "🚪 ВЫХОД", COLORS['ERROR'])
    ]


def create_game_buttons():
    button_width, button_height = 190, 50
    start_x = BOARD_SIZE + MARGIN * 2
    start_y = 700
    return [
        Button(start_x, start_y, button_width, button_height, "🔄 Новая"),
        Button(start_x + 210, start_y, button_width, button_height, "🏠 Меню"),
        Button(start_x, start_y + 70, button_width, button_height, "↩ Отменить"),
        Button(start_x + 210, start_y + 70, button_width, button_height, "🤖 Ход ИИ")
    ]


def create_settings_buttons():
    button_width, button_height = 360, 55
    start_x = (SCREEN_WIDTH - button_width) // 2
    return [
        Button(start_x, 250, button_width, button_height, "⭐ ЛЁГКИЙ (глубина 2)"),
        Button(start_x, 325, button_width, button_height, "⚡ СРЕДНИЙ (глубина 3)"),
        Button(start_x, 400, button_width, button_height, "🔥 СЛОЖНЫЙ (глубина 4)"),
        Button(start_x, 475, button_width, button_height, "👑 ЭКСПЕРТ (глубина 5)"),
        Button(start_x, 580, button_width, button_height, "◀ НАЗАД")
    ]


menu_buttons = create_menu_buttons()
game_buttons = create_game_buttons()
settings_buttons = create_settings_buttons()

# Индикаторы
progress_indicator = ProgressIndicator(BOARD_SIZE + MARGIN * 2, 220, 400, 25)


def draw_gradient_background():
    """Рисует градиентный фон"""
    for y in range(SCREEN_HEIGHT):
        color = (
            COLORS['BACKGROUND'][0] + int(y * 0.02),
            COLORS['BACKGROUND'][1] + int(y * 0.01),
            COLORS['BACKGROUND'][2] + int(y * 0.03)
        )
        pygame.draw.line(screen, color, (0, y), (SCREEN_WIDTH, y))


def draw_board_with_coordinates():
    """Рисует доску с координатами"""
    # Доска
    for row in range(8):
        for col in range(8):
            x = col * SQUARE_SIZE + MARGIN
            y = row * SQUARE_SIZE + MARGIN + 50

            color = COLORS['BOARD_LIGHT'] if (row + col) % 2 == 0 else COLORS['BOARD_DARK']
            pygame.draw.rect(screen, color, (x, y, SQUARE_SIZE, SQUARE_SIZE))

            # Координаты (только по краям)
            if col == 0:
                num = str(8 - row)
                coord = FONTS['SMALL'].render(num, True,
                                              COLORS['TEXT'] if row % 2 == 1 else COLORS['BOARD_DARK'])
                screen.blit(coord, (x + 5, y + 5))

            if row == 7:
                letter = chr(97 + col)
                coord = FONTS['SMALL'].render(letter, True,
                                              COLORS['TEXT'] if col % 2 == 0 else COLORS['BOARD_LIGHT'])
                screen.blit(coord, (x + SQUARE_SIZE - 18, y + SQUARE_SIZE - 22))

    # Подсветка последнего хода
    if last_move:
        from_row = 7 - chess.square_rank(last_move.from_square)
        from_col = chess.square_file(last_move.from_square)
        to_row = 7 - chess.square_rank(last_move.to_square)
        to_col = chess.square_file(last_move.to_square)

        for row, col in [(from_row, from_col), (to_row, to_col)]:
            x = col * SQUARE_SIZE + MARGIN
            y = row * SQUARE_SIZE + MARGIN + 50
            s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            s.fill(COLORS['LAST_MOVE'])
            screen.blit(s, (x, y))

    # Подсветка выбранной фигуры
    if selected_square is not None:
        row = 7 - chess.square_rank(selected_square)
        col = chess.square_file(selected_square)
        x = col * SQUARE_SIZE + MARGIN
        y = row * SQUARE_SIZE + MARGIN + 50
        pygame.draw.rect(screen, COLORS['HIGHLIGHT'], (x, y, SQUARE_SIZE, SQUARE_SIZE), 4)


def draw_pieces_with_shadow():
    """Рисует фигуры с тенями для лучшей видимости"""
    for row in range(8):
        for col in range(8):
            square_idx = chess.square(col, 7 - row)
            piece = board.piece_at(square_idx)

            if piece:
                symbol = piece.symbol()
                if symbol in PIECE_SYMBOLS:
                    emoji = PIECE_SYMBOLS[symbol]
                    x = col * SQUARE_SIZE + MARGIN + SQUARE_SIZE // 2
                    y = row * SQUARE_SIZE + MARGIN + 50 + SQUARE_SIZE // 2

                    # Тень
                    shadow = FONTS['PIECE'].render(emoji, True, (0, 0, 0, 180))
                    shadow_rect = shadow.get_rect(center=(x + 2, y + 2))
                    screen.blit(shadow, shadow_rect)

                    # Фигура (белые или чёрные)
                    color = COLORS['TEXT'] if symbol.isupper() else (20, 20, 20)
                    text = FONTS['PIECE'].render(emoji, True, color)
                    text_rect = text.get_rect(center=(x, y))
                    screen.blit(text, text_rect)


def draw_legal_moves_highlight():
    """Подсветка возможных ходов"""
    if selected_square is not None:
        for move in legal_moves:
            if move.from_square == selected_square:
                row = 7 - chess.square_rank(move.to_square)
                col = chess.square_file(move.to_square)
                center_x = col * SQUARE_SIZE + MARGIN + SQUARE_SIZE // 2
                center_y = row * SQUARE_SIZE + MARGIN + 50 + SQUARE_SIZE // 2

                if board.piece_at(move.to_square):
                    # Взятие - красный кружок
                    pygame.draw.circle(screen, (255, 80, 80, 220),
                                       (center_x, center_y), SQUARE_SIZE // 3, 4)
                else:
                    # Обычный ход - зелёный кружок
                    pygame.draw.circle(screen, COLORS['LEGAL_MOVE'][:3],
                                       (center_x, center_y), SQUARE_SIZE // 6)


def safe_san(board_state, move):
    """Безопасное получение SAN нотации с проверкой легальности"""
    try:
        if move in board_state.legal_moves:
            return board_state.san(move)
        else:
            # Пытаемся получить UCI нотацию
            return chess.square_name(move.from_square) + chess.square_name(move.to_square)
    except:
        return chess.square_name(move.from_square) + chess.square_name(move.to_square)


def draw_info_panel():
    """Правая панель с информацией"""
    panel_x = BOARD_SIZE + MARGIN * 2
    panel_width = SCREEN_WIDTH - panel_x - MARGIN

    # Фон панели
    pygame.draw.rect(screen, COLORS['PANEL_BG'],
                     (panel_x, MARGIN, panel_width, SCREEN_HEIGHT - MARGIN * 2),
                     border_radius=15)

    y_offset = MARGIN + 20

    # Заголовок
    title = FONTS['HEADER'].render("ШАХМАТЫ AI", True, COLORS['ACCENT'])
    screen.blit(title, (panel_x + (panel_width - title.get_width()) // 2, y_offset))
    y_offset += 60

    # Статус ИИ
    ai_status = FONTS['INFO'].render("✅ Python Chess AI готов", True, COLORS['SUCCESS'])
    screen.blit(ai_status, (panel_x + 20, y_offset))
    y_offset += 40

    # Сложность
    diff_names = ["ЛЁГКИЙ", "СРЕДНИЙ", "СЛОЖНЫЙ", "ЭКСПЕРТ"]
    diff_text = f"Сложность: {diff_names[difficulty - 1]}"
    diff = FONTS['INFO'].render(diff_text, True, COLORS['TEXT'])
    screen.blit(diff, (panel_x + 20, y_offset))
    y_offset += 40

    # Чей ход
    turn_text = "ХОД БЕЛЫХ" if board.turn == chess.WHITE else "ХОД ЧЁРНЫХ"
    turn_color = COLORS['TEXT'] if board.turn == chess.WHITE else (200, 200, 200)
    turn_bg = (70, 80, 100) if board.turn == chess.WHITE else (50, 60, 80)

    turn_rect = pygame.Rect(panel_x + 20, y_offset, panel_width - 40, 45)
    pygame.draw.rect(screen, turn_bg, turn_rect, border_radius=10)
    pygame.draw.rect(screen, COLORS['ACCENT'], turn_rect, 3, border_radius=10)

    turn = FONTS['BUTTON'].render(turn_text, True, turn_color)
    screen.blit(turn, (turn_rect.centerx - turn.get_width() // 2,
                       turn_rect.centery - turn.get_height() // 2))
    y_offset += 70

    # Статус игры
    if board.is_checkmate():
        status = "♔ МАТ!"
        color = COLORS['ERROR']
    elif board.is_stalemate():
        status = "═ ПАТ"
        color = COLORS['WARNING']
    elif board.is_check():
        status = "⚡ ШАХ!"
        color = COLORS['ERROR']
    elif board.is_game_over():
        status = "■ ИГРА ОКОНЧЕНА"
        color = (150, 150, 150)
    else:
        status = "▶ ИГРА ИДЁТ"
        color = COLORS['SUCCESS']

    game_status = FONTS['INFO'].render(status, True, color)
    screen.blit(game_status, (panel_x + 20, y_offset))
    y_offset += 60

    # Индикатор прогресса ИИ
    progress_indicator.rect.x = panel_x + 20
    progress_indicator.rect.y = y_offset
    progress_indicator.rect.width = panel_width - 40

    think_time = time.time() - think_start_time if is_thinking else 0
    progress_indicator.update(is_thinking, think_time)
    progress_indicator.draw(screen, is_thinking, thinking_depth)
    y_offset += 80

    # История ходов
    moves_title = FONTS['INFO'].render("ПОСЛЕДНИЕ ХОДЫ:", True, COLORS['ACCENT'])
    screen.blit(moves_title, (panel_x + 20, y_offset))
    y_offset += 35

    # Отображаем ходы в две колонки
    moves = list(board.move_stack)
    col1_x = panel_x + 25
    col2_x = panel_x + panel_width // 2 + 10
    max_rows = 8

    for i in range(0, min(len(moves), max_rows * 2), 2):
        move_num = i // 2 + 1
        col = col1_x if (i // 2) % 2 == 0 else col2_x
        row = ((i // 2) % max_rows) * 28

        if i < len(moves):
            try:
                white_move = safe_san(board, moves[i])
            except:
                white_move = "??"

            move_text = f"{move_num:2d}. {white_move:6s}"

            if i + 1 < len(moves):
                try:
                    black_move = safe_san(board, moves[i + 1])
                except:
                    black_move = "??"
                move_text += f"  {black_move:6s}"

            move_surf = FONTS['SMALL'].render(move_text, True, COLORS['TEXT'])

            # Проверяем, чтобы текст не выходил за границы панели
            if y_offset + row < SCREEN_HEIGHT - 100:
                screen.blit(move_surf, (col, y_offset + row))

    # Кнопки управления
    for btn in game_buttons:
        btn.draw(screen)

    # Статусное сообщение
    if status_message:
        status_surf = FONTS['SMALL'].render(status_message, True, status_color)
        screen.blit(status_surf, (panel_x + 20, SCREEN_HEIGHT - MARGIN - 35))


def draw_menu_screen():
    """Главное меню"""
    draw_gradient_background()

    # Заголовок с иконками
    title1 = FONTS['TITLE'].render("♔ ШАХМАТЫ", True, COLORS['ACCENT'])
    title2 = FONTS['TITLE'].render("PYTHON AI ♚", True, COLORS['TEXT'])

    screen.blit(title1, ((SCREEN_WIDTH - title1.get_width()) // 2, 150))
    screen.blit(title2, ((SCREEN_WIDTH - title2.get_width()) // 2, 220))

    # Подзаголовок
    subtitle = FONTS['INFO'].render("Игра против искусственного интеллекта на Python",
                                    True, (180, 200, 255))
    screen.blit(subtitle, ((SCREEN_WIDTH - subtitle.get_width()) // 2, 290))

    # Кнопки меню
    for btn in menu_buttons:
        btn.draw(screen)

    # Статус
    ai_status = FONTS['INFO'].render("✅ Python Chess AI готов к игре",
                                     True, COLORS['SUCCESS'])
    screen.blit(ai_status, ((SCREEN_WIDTH - ai_status.get_width()) // 2, 680))

    # Подсказка
    hint = FONTS['SMALL'].render("Не требует установки Stockfish • Работает на чистом Python",
                                 True, (150, 180, 220))
    screen.blit(hint, ((SCREEN_WIDTH - hint.get_width()) // 2, 730))


def draw_settings_screen():
    """Экран настроек"""
    draw_gradient_background()

    title = FONTS['HEADER'].render("⚙ ВЫБОР СЛОЖНОСТИ ИИ", True, COLORS['ACCENT'])
    screen.blit(title, ((SCREEN_WIDTH - title.get_width()) // 2, 120))

    desc = FONTS['INFO'].render("Глубина анализа определяет силу искусственного интеллекта:",
                                True, COLORS['TEXT'])
    screen.blit(desc, ((SCREEN_WIDTH - desc.get_width()) // 2, 180))

    # Кнопки сложности
    for btn in settings_buttons:
        # Подсвечиваем текущую сложность
        prefixes = ["⭐", "⚡", "🔥", "👑"]
        if btn.text.startswith(prefixes[difficulty - 1]):
            btn.hovered = True
        btn.draw(screen)

    # Текущая настройка
    diff_names = ["ЛЁГКИЙ", "СРЕДНИЙ", "СЛОЖНЫЙ", "ЭКСПЕРТ"]
    current_text = f"Текущая сложность: {diff_names[difficulty - 1]}"
    current = FONTS['INFO'].render(current_text, True, COLORS['SUCCESS'])
    screen.blit(current, ((SCREEN_WIDTH - current.get_width()) // 2, 630))

    # Подсказка
    hint = FONTS['SMALL'].render("Более глубкий анализ = сильнее ИИ = дольше время хода",
                                 True, (150, 180, 220))
    screen.blit(hint, ((SCREEN_WIDTH - hint.get_width()) // 2, 680))


def draw_game_over_screen():
    """Экран окончания игры"""
    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 220))
    screen.blit(overlay, (0, 0))

    if board.is_checkmate():
        winner = "БЕЛЫХ" if not board.turn else "ЧЁРНЫХ"
        result_text = f"♔ МАТ! ПОБЕДИЛИ {winner} ♚"
        color = (255, 215, 0)
    elif board.is_stalemate():
        result_text = "═ ПАТ - НИЧЬЯ ═"
        color = (200, 200, 100)
    elif board.is_insufficient_material():
        result_text = "■ НЕДОСТАТОЧНО ФИГУР ■"
        color = (150, 150, 150)
    else:
        result_text = "■ ИГРА ОКОНЧЕНА ■"
        color = (150, 150, 150)

    result = FONTS['TITLE'].render(result_text, True, color)
    screen.blit(result, ((SCREEN_WIDTH - result.get_width()) // 2,
                         SCREEN_HEIGHT // 2 - 60))

    restart = FONTS['INFO'].render("Нажмите N для новой игры • ESC для выхода в меню",
                                   True, COLORS['TEXT'])
    screen.blit(restart, ((SCREEN_WIDTH - restart.get_width()) // 2,
                          SCREEN_HEIGHT // 2 + 40))


def start_new_game(color):
    """Начинает новую игру"""
    global board, selected_square, legal_moves, last_move, game_over, player_color
    global status_message, status_color, ai_move_history

    board = chess.Board()
    selected_square = None
    legal_moves = []
    last_move = None
    game_over = False
    player_color = color
    ai_move_history = []

    status_message = f"Новая игра: вы играете за {'белых' if color == chess.WHITE else 'чёрных'}"
    status_color = COLORS['SUCCESS']

    print(f"\n{'=' * 60}")
    print(f"НОВАЯ ИГРА: Вы играете за {'белых' if color == chess.WHITE else 'чёрных'}")
    print(f"{'=' * 60}")

    # Если играем чёрными, ИИ ходит первым
    if color == chess.BLACK:
        make_ai_move()


def make_ai_move():
    """Запускает ход ИИ"""
    global is_thinking, think_start_time, status_message, status_color

    if board.is_game_over() or is_thinking:
        return

    is_thinking = True
    think_start_time = time.time()
    status_message = "🤖 Python AI анализирует позицию..."
    status_color = COLORS['ACCENT']

    threading.Thread(target=_ai_move_thread, daemon=True).start()


def _ai_move_thread():
    """Фоновый поток для хода ИИ"""
    global is_thinking, status_message, status_color

    try:
        start_time = time.time()

        # Создаём копию доски для безопасного анализа
        board_copy = board.copy()
        move = ai_engine.get_best_move(board_copy, difficulty)
        think_time = time.time() - start_time

        if move and move in board.legal_moves:
            # Сохраняем ход для выполнения в основном потоке
            ai_engine.pending_move = move

            # Получаем нотацию хода безопасно
            try:
                move_san = board.san(move)
            except:
                move_san = f"{chess.square_name(move.from_square)}-{chess.square_name(move.to_square)}"

            status_message = f"✅ AI: {move_san} (за {think_time:.1f}с)"
            status_color = COLORS['SUCCESS']
            print(f"Python AI: {move_san} (за {think_time:.2f}с)")

            # Сохраняем в историю
            ai_move_history.append((move_san, think_time))
        else:
            status_message = "⚠ AI не смог найти легальный ход"
            status_color = COLORS['ERROR']

    except Exception as e:
        print(f"Ошибка AI: {e}")
        status_message = f"⚠ Ошибка AI: {str(e)[:50]}"
        status_color = COLORS['ERROR']

    is_thinking = False


def handle_board_click(pos):
    """Обработка кликов по доске"""
    global selected_square, legal_moves, last_move, game_over
    global status_message, status_color

    if game_over or is_thinking or board.turn != player_color:
        if is_thinking:
            status_message = "⏳ Дождитесь хода AI..."
            status_color = COLORS['WARNING']
        return

    x, y = pos
    board_y_start = MARGIN + 50

    if not (MARGIN <= x < MARGIN + BOARD_SIZE and
            board_y_start <= y < board_y_start + BOARD_SIZE):
        return

    col = (x - MARGIN) // SQUARE_SIZE
    row = (y - board_y_start) // SQUARE_SIZE
    square_idx = chess.square(col, 7 - row)

    if selected_square is not None:
        # Пытаемся сделать ход
        for move in legal_moves:
            if move.from_square == selected_square and move.to_square == square_idx:
                try:
                    board.push(move)
                    last_move = move
                    selected_square = None
                    legal_moves = []
                    game_over = board.is_game_over()

                    if not game_over and board.turn != player_color:
                        make_ai_move()
                    return
                except Exception as e:
                    status_message = f"⚠ Нелегальный ход: {str(e)[:30]}"
                    status_color = COLORS['ERROR']
                    selected_square = None
                    legal_moves = []
                    return

        # Выбор другой своей фигуры
        piece = board.piece_at(square_idx)
        if piece and piece.color == player_color:
            selected_square = square_idx
            legal_moves = [m for m in board.legal_moves if m.from_square == square_idx]
        else:
            selected_square = None
            legal_moves = []
    else:
        # Выбор фигуры
        piece = board.piece_at(square_idx)
        if piece and piece.color == player_color:
            selected_square = square_idx
            legal_moves = [m for m in board.legal_moves if m.from_square == square_idx]


# Основной игровой цикл
def main():
    global current_state, difficulty, game_over, player_color, is_thinking
    global status_message, status_color

    running = True
    ai_engine.pending_move = None

    print("\n" + "=" * 60)
    print("ШАХМАТЫ PYTHON AI - ЗАПУСК")
    print("=" * 60)
    print("✅ Используется чистый Python AI (без Stockfish)")
    print("✅ Исправлены ошибки обработки ходов")
    print("=" * 60)

    while running:
        mouse_pos = pygame.mouse.get_pos()

        # Обновляем hover состояние кнопок
        if current_state == "MENU":
            for btn in menu_buttons:
                btn.check_hover(mouse_pos)
        elif current_state == "PLAYING":
            for btn in game_buttons:
                btn.check_hover(mouse_pos)
        elif current_state == "SETTINGS":
            for btn in settings_buttons:
                btn.check_hover(mouse_pos)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if current_state == "MENU":
                    for btn in menu_buttons:
                        if btn.is_clicked(mouse_pos, event.type):
                            if "БЕЛЫМИ" in btn.text:
                                start_new_game(chess.WHITE)
                                current_state = "PLAYING"
                            elif "ЧЁРНЫМИ" in btn.text:
                                start_new_game(chess.BLACK)
                                current_state = "PLAYING"
                            elif "НАСТРОЙКИ" in btn.text:
                                current_state = "SETTINGS"
                            elif "ВЫХОД" in btn.text:
                                running = False

                elif current_state == "PLAYING":
                    btn_clicked = False
                    for btn in game_buttons:
                        if btn.is_clicked(mouse_pos, event.type):
                            btn_clicked = True
                            if "Новая" in btn.text:
                                start_new_game(player_color)
                            elif "Меню" in btn.text:
                                current_state = "MENU"
                                status_message = ""
                            elif "Отменить" in btn.text:
                                if len(board.move_stack) > 0:
                                    board.pop()
                                    if len(board.move_stack) > 0 and board.turn != player_color:
                                        board.pop()
                                    selected_square = None
                                    legal_moves = []
                                    game_over = False
                                    is_thinking = False
                                    ai_engine.pending_move = None
                                    status_message = "↩ Ход отменён"
                                    status_color = COLORS['ACCENT']
                            elif "Ход ИИ" in btn.text:
                                if not is_thinking and board.turn != player_color:
                                    make_ai_move()

                    if not btn_clicked:
                        handle_board_click(mouse_pos)

                elif current_state == "SETTINGS":
                    for btn in settings_buttons:
                        if btn.is_clicked(mouse_pos, event.type):
                            if "ЛЁГКИЙ" in btn.text:
                                difficulty = 1
                                status_message = "✅ Установлена лёгкая сложность (глубина 2)"
                                status_color = COLORS['SUCCESS']
                            elif "СРЕДНИЙ" in btn.text:
                                difficulty = 2
                                status_message = "✅ Установлена средняя сложность (глубина 3)"
                                status_color = COLORS['SUCCESS']
                            elif "СЛОЖНЫЙ" in btn.text:
                                difficulty = 3
                                status_message = "✅ Установлена сложная сложность (глубина 4)"
                                status_color = COLORS['SUCCESS']
                            elif "ЭКСПЕРТ" in btn.text:
                                difficulty = 4
                                status_message = "✅ Установлена экспертная сложность (глубина 5)"
                                status_color = COLORS['SUCCESS']
                            elif "НАЗАД" in btn.text:
                                current_state = "MENU"
                                status_message = ""

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if current_state == "PLAYING":
                        current_state = "MENU"
                        status_message = ""
                    elif current_state == "SETTINGS":
                        current_state = "MENU"
                        status_message = ""
                    else:
                        running = False
                elif event.key == pygame.K_n and current_state == "PLAYING":
                    start_new_game(player_color)

        # Обработка хода AI
        if current_state == "PLAYING" and ai_engine.pending_move and not is_thinking:
            try:
                if ai_engine.pending_move in board.legal_moves:
                    board.push(ai_engine.pending_move)
                    last_move = ai_engine.pending_move
                    game_over = board.is_game_over()
                else:
                    status_message = "⚠ AI предложил нелегальный ход"
                    status_color = COLORS['ERROR']
            except Exception as e:
                status_message = f"⚠ Ошибка выполнения хода AI: {str(e)[:30]}"
                status_color = COLORS['ERROR']

            ai_engine.pending_move = None

        # Отрисовка
        screen.fill(COLORS['BACKGROUND'])

        if current_state == "MENU":
            draw_menu_screen()

        elif current_state == "PLAYING":
            draw_board_with_coordinates()
            draw_legal_moves_highlight()
            draw_pieces_with_shadow()
            draw_info_panel()

            if game_over:
                draw_game_over_screen()

        elif current_state == "SETTINGS":
            draw_settings_screen()

        # Обновляем индикатор прогресса
        progress_indicator.update(is_thinking)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()