import pygame
import sys
import random
import math
import os
import time

# 获取资源路径（兼容 pyinstaller 打包后的 exe）
def get_path(relative_path):
    if getattr(sys, 'frozen', False):  # 如果是打包后的 exe
        base_path = sys._MEIPASS
    else:  # 普通运行
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ==================== 常量定义（可在导入时使用）====================
WIDTH, HEIGHT = 800, 600

# 定义颜色
GRAY = (128,128,128)
BLACK = (0,0,0)
RED = (255,0,0)
BLUE = (0,0,255)
WHITE = (255,255,255)
YELLOW = (255,255,0)
CYAN = (0,255,255)
MAGENTA = (255, 0, 255)

# 小球参数
player_radius = 15
player_speed = 4
player_x, player_y = WIDTH // 2, HEIGHT // 2

# 敌人参数
enemies = []
enemy_size = 30
enemy_speed = 2
spawn_interval = 1500
spawn_event = pygame.USEREVENT + 1  # 事件常量，不需要 pygame 初始化

# 波纹攻击参数
waves = []
normal_wave_speed = 3
charged_wave_speed = 5
normal_wave_radius = 75
charged_wave_radius = 150

# 蓄力相关
charging = False
charge_canceled = False
charge_start_time = 0
charge_max_radius = 50
charge_max_time = 0.5
charge_max_start_time = 0
charge_cancel_played = False

# 分数（初始值，会被重置）
score = 0
show_double = False
double_timer = 0
double_wave_hits = 0

# 游戏结束标志
game_over = False
game_over_played = False

# ==================== 主程序（仅在直接运行时执行）====================
if __name__ == "__main__":
    # 初始化pygame
    pygame.init()
    pygame.mixer.init()

    # 加载背景音乐
    pygame.mixer.music.load(get_path("sounds/bgm_piano_converted.wav"))
    pygame.mixer.music.set_volume(0.3)
    pygame.mixer.music.play(-1)  # 循环播放

    # 加载音效
    attack_sound = pygame.mixer.Sound(get_path("sounds/attack.wav"))
    charge_cancel_sound = pygame.mixer.Sound(get_path("sounds/charge_cancel.wav"))
    charged_attack_sound = pygame.mixer.Sound(get_path("sounds/charged_attack.wav"))

    pop_folder = get_path("sounds/pop_parts")
    pop_sounds = [pygame.mixer.Sound(os.path.join(pop_folder, f)) for f in os.listdir(pop_folder)]

    def play_random_pop():
        random.choice(pop_sounds).play()

    # 创建窗口
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("raindrol")

    # 设置定时器
    pygame.time.set_timer(spawn_event, spawn_interval)

    # 字体（需要 pygame 初始化后才能创建）
    font_small = pygame.font.SysFont(None, 24)   # 分数
    font_medium = pygame.font.SysFont(None, 32)  # Double!
    font_large = pygame.font.SysFont(None, 48)   # Game Over

    # 游戏循环
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN and game_over:
                    # 重置游戏状态
                    player_x, player_y = WIDTH // 2, HEIGHT // 2
                    enemies = []
                    waves = []
                    score = 0
                    game_over = False

            if not game_over:
                if event.type == spawn_event:
                    side = random.choice(["top", "bottom", "left", "right"])
                    if side == "top":
                        x, y = random.randint(0, WIDTH), 0
                    elif side == "bottom":
                        x, y = random.randint(0, WIDTH), HEIGHT
                    elif side == "left":
                        x, y = 0, random.randint(0, HEIGHT)
                    else:
                        x, y = WIDTH, random.randint(0, HEIGHT)
                    enemies.append([x,y])

                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    if not charging:
                        charging = True
                        charge_start_time = time.time()

                if event.type == pygame.KEYUP and event.key == pygame.K_SPACE:
                    if charging and not charge_canceled:
                        charging = False
                        hold_time = time.time() - charge_start_time
                        if hold_time >= 1:
                            waves.append([player_x, player_y, 0, charged_wave_radius, 0, charged_wave_speed])
                            charged_attack_sound.play()
                        else:
                            waves.append([player_x, player_y, 0, normal_wave_radius, 0, normal_wave_speed])
                            attack_sound.play()
                    charge_canceled = False
                    charge_cancel_played = False

        if not game_over:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT] and player_x - player_radius > 0:
                player_x -= player_speed
            if keys[pygame.K_RIGHT] and player_x + player_radius < WIDTH:
                player_x += player_speed
            if keys[pygame.K_UP] and player_y - player_radius > 0:
                player_y -= player_speed
            if keys[pygame.K_DOWN] and player_y + player_radius < HEIGHT:
                player_y += player_speed

            for enemy in enemies:
                dx = player_x - enemy[0]
                dy = player_y - enemy[1]
                dist = math.hypot(dx, dy)
                if dist != 0:
                    enemy[0] += enemy_speed * dx / dist
                    enemy[1] += enemy_speed * dy / dist

            new_interval = max(500, 1500 - (score // 10) * 100)
            if new_interval != spawn_interval:
                spawn_interval = new_interval
                pygame.time.set_timer(spawn_event, spawn_interval)

            for enemy in enemies:
                dist = math.hypot(player_x - enemy[0], player_y - enemy[1])
                if dist < player_radius + enemy_size // 2:
                    game_over = True

            new_waves = []
            for wave in waves:
                wave[2] += wave[5]
                if wave[2] < wave[3]:
                    new_waves.append(wave)
                else:
                    if wave[4] > 0:
                        double_wave_hits = wave[4]
                        if wave[4] > 1:
                            score += wave[4] * 2
                            show_double = True
                            double_timer = pygame.time.get_ticks()
                        else:
                            score += wave[4]
            waves = new_waves

            new_enemies = []
            for enemy in enemies:
                hit = False
                for wave in waves:
                    dist = math.hypot(enemy[0] - wave[0], enemy[1] - wave[1])
                    if abs(dist - wave[2]) < 10:
                        wave[4] += 1
                        hit = True
                        play_random_pop()
                        break
                if not hit:
                    new_enemies.append(enemy)
            enemies = new_enemies

        screen.fill(GRAY)
        if not game_over:
            pygame.draw.circle(screen, BLACK, (player_x, player_y), player_radius)

            if charging:
                charge_time = time.time() - charge_start_time
                charge_radius = int(charge_time * 50)
                if charge_radius >= charge_max_radius:
                    charge_radius = charge_max_radius
                    if charge_max_start_time == 0:
                        charge_max_start_time = time.time()
                    if time.time() - charge_max_start_time < charge_max_time:
                        color = RED
                    else:
                        charging = False
                        charge_canceled = True
                        charge_max_start_time = 0
                        if not charge_cancel_played:
                            charge_cancel_sound.play()
                            charge_cancel_played = True
                        color = MAGENTA
                else:
                    charge_max_start_time = 0
                    charge_cancel_played = False
                    color = CYAN
                pygame.draw.circle(screen, color, (player_x, player_y), player_radius + charge_radius, 2)

            for enemy in enemies:
                pygame.draw.rect(screen, RED, (enemy[0] - enemy_size//2, enemy[1] - enemy_size//2, enemy_size, enemy_size))

            for wave in waves:
                pygame.draw.circle(screen, BLUE, (wave[0], wave[1]), wave[2], 2)

            score_text = font_small.render(f"Score: {score}", True, BLACK)
            screen.blit(score_text, (10, 10))

            if show_double:
                if pygame.time.get_ticks() - double_timer < 1000:
                    double_text = font_medium.render(f"{double_wave_hits} COMBO!", True, YELLOW)
                    screen.blit(double_text, (WIDTH - 220, 100))
                else:
                    show_double = False
        else:
            if not game_over_played:
                charge_cancel_sound.play()
                game_over_played = True

            game_over_text = font_large.render("GAME OVER", True, BLACK)
            restart_text = font_medium.render("Press Enter to Restart", True, WHITE)
            score_text = font_small.render(f"Your Score: {score}", True, BLACK)

            screen.blit(game_over_text, (WIDTH // 2 - game_over_text.get_width() // 2,
                                         HEIGHT // 2 - game_over_text.get_height() // 2 - 30))
            screen.blit(restart_text, (WIDTH // 2 - restart_text.get_width() // 2,
                                       HEIGHT // 2 - restart_text.get_height() // 2 + 30))
            screen.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, HEIGHT // 2 - 100))

        pygame.display.flip()
        clock.tick(60)