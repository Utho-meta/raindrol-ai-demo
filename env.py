import pygame
import numpy as np
import random
import math
import time
from game import *  # 导入原游戏中的常量、函数等


class RaindrolEnv:
    def __init__(self, render_mode=False):
        # Pygame 初始化（如果多次初始化可能出问题，这里用标志避免重复）
        if not pygame.get_init():
            pygame.init()
        self.width, self.height = WIDTH, HEIGHT
        self.render_mode = render_mode
        if render_mode:
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Raindrol RL Training")
            self.clock = pygame.time.Clock()
        else:
            self.screen = None

        # 游戏参数（直接使用原游戏中的）
        self.player_radius = player_radius
        self.player_speed = player_speed
        self.enemy_size = enemy_size
        self.enemy_speed = enemy_speed
        self.normal_wave_radius = normal_wave_radius
        self.charged_wave_radius = charged_wave_radius
        self.normal_wave_speed = normal_wave_speed
        self.charged_wave_speed = charged_wave_speed

        # 限制最大敌人数（避免敌人无限增长）
        self.max_enemies = 30

        # 奖励临时变量
        self.charged_attack_bonus = 0.0  # 蓄力攻击奖励次数
        self.combo_bonus = 0.0           # 连杀奖励次数
        self.attack_penalty = 0.0         # 攻击惩罚次数
        self.hit_bonus = 0.0              # 击中奖励（暂未使用）

        # 状态变量
        self.reset()

    def reset(self):
        """重置游戏，返回初始状态"""
        self.player_x = WIDTH // 2
        self.player_y = HEIGHT // 2
        self.enemies = []  # 每个元素 [x, y]
        self.waves = []  # 每个元素 [x, y, current_radius, max_radius, hit_count, speed]
        self.score = 0
        self.charging = False
        self.charge_start_time = 0
        self.charge_canceled = False
        self.game_over = False

        # 记录上一帧得分用于奖励计算
        self.last_score = 0

        # 重置统计
        self.stats_attacks = 0
        self.stats_kills = 0
        self.stats_combos = 0
        self.steps_this_episode = 0
        self.last_kills = 0

        # 清空 Pygame 事件队列，避免残留
        pygame.event.clear()

        return self._get_state()

    def _get_state(self):
        """
        构造状态向量。
        我们设计一个固定长度的向量，包含：
        - 玩家归一化坐标 (2)
        - 最近 K 个敌人的归一化坐标 (K*2)，K=10，不足补0
        - 蓄力进度 (0~1)
        - 当前分数归一化 (1)
        总共 2 + 20 + 1 + 1 = 24 维
        """
        K = 10
        # 计算所有敌人到玩家的距离，取最近的K个
        enemy_dists = []
        for ex, ey in self.enemies:
            dist = math.hypot(ex - self.player_x, ey - self.player_y)
            enemy_dists.append((dist, ex, ey))
        enemy_dists.sort(key=lambda x: x[0])

        # 填充坐标，归一化到 [0,1]
        enemy_pos = np.zeros((K, 2))
        for i, (_, ex, ey) in enumerate(enemy_dists[:K]):
            enemy_pos[i] = [ex / self.width, ey / self.height]

        # 蓄力进度
        charge_progress = 0.0
        if self.charging:
            hold_time = time.time() - self.charge_start_time
            charge_progress = min(hold_time / 1.0, 1.0)  # 1秒满蓄力

        # 分数归一化（假设最大可能分数 1000，可调整）
        score_norm = min(self.score / 1000, 1.0)

        state = np.concatenate([
            [self.player_x / self.width, self.player_y / self.height],
            enemy_pos.flatten(),
            [charge_progress],
            [score_norm]
        ])
        return state.astype(np.float32)

    def step(self, action):
        """
        执行动作。
        动作空间：我们定义为 18 个离散动作，对应 9 种移动方向 × 2 种攻击状态。
        移动方向编码：0=不动,1=上,2=下,3=左,4=右,5=左上,6=右上,7=左下,8=右下
        攻击状态：0=不攻击（松开空格），1=攻击（按下空格）
        动作索引 = 移动方向 * 2 + 攻击状态
        """
        # 解析动作
        move_action = action // 2  # 0-8
        attack_action = action % 2  # 0-1

        # 处理移动
        dx, dy = 0, 0
        if move_action == 1:
            dy = -1
        elif move_action == 2:
            dy = 1
        elif move_action == 3:
            dx = -1
        elif move_action == 4:
            dx = 1
        elif move_action == 5:
            dx, dy = -1, -1
        elif move_action == 6:
            dx, dy = 1, -1
        elif move_action == 7:
            dx, dy = -1, 1
        elif move_action == 8:
            dx, dy = 1, 1

        old_x, old_y = self.player_x, self.player_y  # 记录移动前坐标
        new_x = self.player_x + dx * self.player_speed
        new_y = self.player_y + dy * self.player_speed
        # 边界检查
        if self.player_radius <= new_x <= self.width - self.player_radius:
            self.player_x = new_x
        if self.player_radius <= new_y <= self.height - self.player_radius:
            self.player_y = new_y

        # 处理攻击（空格）
        if attack_action == 1:
            # 按下空格
            if not self.charging:
                self.charging = True
                self.charge_start_time = time.time()
        else:
            # 松开空格
            if self.charging and not self.charge_canceled:
                self.charging = False
                hold_time = time.time() - self.charge_start_time
                if hold_time >= 1.0:
                    self.waves.append([self.player_x, self.player_y, 0,
                                       self.charged_wave_radius, 0, self.charged_wave_speed])
                    # 蓄力攻击奖励 +1 次
                    self.charged_attack_bonus += 1.0
                    # 攻击惩罚 +1 次
                    self.attack_penalty += 1.0
                    self.stats_attacks += 1
                else:
                    self.waves.append([self.player_x, self.player_y, 0,
                                       self.normal_wave_radius, 0, self.normal_wave_speed])
                    # 攻击惩罚 +1 次
                    self.attack_penalty += 1.0
                    self.stats_attacks += 1
            self.charge_canceled = False

        # 更新游戏逻辑（敌人移动、波纹扩散、碰撞等）
        self._update_game()

        # 计算奖励
        reward = self._compute_reward()

        # 步数统计（仅用于打印，不影响游戏结束）
        self.steps_this_episode += 1

        # 检查是否结束（只有玩家与敌人碰撞才会触发 game_over）
        done = self.game_over

        # 获取下一状态
        next_state = self._get_state()

        # 可选渲染
        if self.render_mode:
            self._render()

        return next_state, reward, done, {}

    def _update_game(self):
        """游戏每帧逻辑（从原主循环提取，稍作调整）"""
        # 1. 敌人追踪玩家
        for enemy in self.enemies:
            dx = self.player_x - enemy[0]
            dy = self.player_y - enemy[1]
            dist = math.hypot(dx, dy)
            if dist != 0:
                enemy[0] += self.enemy_speed * dx / dist
                enemy[1] += self.enemy_speed * dy / dist

        # 2. 生成新敌人（用概率代替定时器）
        if len(self.enemies) < self.max_enemies and random.random() < 0.02:
            side = random.choice(["top", "bottom", "left", "right"])
            if side == "top":
                x, y = random.randint(0, self.width), 0
            elif side == "bottom":
                x, y = random.randint(0, self.width), self.height
            elif side == "left":
                x, y = 0, random.randint(0, self.height)
            else:
                x, y = self.width, random.randint(0, self.height)
            self.enemies.append([x, y])

        # 3. 波纹扩展与消失
        new_waves = []
        for wave in self.waves:
            wave[2] += wave[5]  # current_radius += speed
            if wave[2] < wave[3]:  # 未达到最大半径
                new_waves.append(wave)
            else:
                # 波纹消失，根据击中数加分
                if wave[4] > 0:
                    self.score += wave[4]
                    self.stats_kills += wave[4]               # 记录击杀数
                    if wave[4] > 1:
                        self.stats_combos += 1                 # 记录连杀次数
                        self.combo_bonus += 1.0                # 连杀奖励次数+1
        self.waves = new_waves

        # 4. 敌人与波纹碰撞
        new_enemies = []
        for enemy in self.enemies:
            hit = False
            for wave in self.waves:
                dist = math.hypot(enemy[0] - wave[0], enemy[1] - wave[1])
                if abs(dist - wave[2]) < 10:  # 判断是否在波纹环上
                    wave[4] += 1
                    hit = True
                    break
            if not hit:
                new_enemies.append(enemy)
        self.enemies = new_enemies

        # 5. 玩家与敌人碰撞
        for enemy in self.enemies:
            dist = math.hypot(self.player_x - enemy[0], self.player_y - enemy[1])
            if dist < self.player_radius + self.enemy_size // 2:
                self.game_over = True
                break

    def _compute_reward(self):
        reward = 0.0

        # 1. 击杀奖励：每个敌人 +100 分
        kill_inc = self.stats_kills - self.last_kills
        reward += kill_inc * 100.0
        self.last_kills = self.stats_kills

        # 2. 鼓励生存：每步 +0.02
        reward += 0.02

        # 3. 距离惩罚（如果敌人太近，扣分）
        min_enemy_dist = float('inf')
        for ex, ey in self.enemies:
            dist = math.hypot(self.player_x - ex, self.player_y - ey)
            if dist < min_enemy_dist:
                min_enemy_dist = dist
        if min_enemy_dist < 100:
            reward -= 0.05  # 距离惩罚

        # 4. 蓄力攻击奖励：每次 +10 分
        if self.charged_attack_bonus > 0:
            reward += self.charged_attack_bonus * 10.0
            self.charged_attack_bonus = 0.0

        # 5. 连杀奖励：每次 +50 分
        if self.combo_bonus > 0:
            reward += self.combo_bonus * 50.0
            self.combo_bonus = 0.0

        # 6. 攻击惩罚：每次攻击扣 0.05 分
        if self.attack_penalty > 0:
            reward -= self.attack_penalty * 0.05
            self.attack_penalty = 0.0

        # 7. 死亡惩罚
        if self.game_over:
            reward -= 5.0

        return reward

    def _render(self):
        """渲染当前画面（与原游戏类似）"""
        if not self.render_mode or not self.screen:
            return

        self.screen.fill(GRAY)
        # 玩家
        pygame.draw.circle(self.screen, BLACK, (int(self.player_x), int(self.player_y)), self.player_radius)
        # 敌人
        for ex, ey in self.enemies:
            pygame.draw.rect(self.screen, RED,
                             (ex - self.enemy_size // 2, ey - self.enemy_size // 2,
                              self.enemy_size, self.enemy_size))
        # 波纹
        for wx, wy, cr, mr, _, _ in self.waves:
            pygame.draw.circle(self.screen, BLUE, (int(wx), int(wy)), int(cr), 2)
        # 分数
        font = pygame.font.SysFont(None, 24)
        score_text = font.render(f"Score: {self.score}", True, BLACK)
        self.screen.blit(score_text, (10, 10))
        # 如果蓄力，画蓄力圈
        if self.charging:
            charge_time = time.time() - self.charge_start_time
            charge_radius = int(charge_time * 50)
            if charge_radius >= 50:
                charge_radius = 50
            pygame.draw.circle(self.screen, CYAN, (int(self.player_x), int(self.player_y)),
                               self.player_radius + charge_radius, 2)

        pygame.display.flip()
        self.clock.tick(15)  # 保持低帧率

    def close(self):
        if self.screen is not None:
            pygame.quit()