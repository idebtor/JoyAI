#pygame 과 NEAT-python 알고리즘을 사용하여 AI가 스스로 플래피 버드 게임 방법을 터득하여 높은 점수를 얻는 프로젝트입니다.

#Tech With Tim
#유튜브 채널 https://www.youtube.com/channel/UC4JX40jDee_tINbkjycV4Sg

#먼저 시작하기 전에 프로그램 실행에 필요한 모듈들을 import 합니다.
import pygame
import random
import os
import time
import neat
import visualize
import pickle
pygame.font.init()  #폰트 모듈을 초기화 합니다.

#플래피 버드 게임 윈도우 스크린 설정, 해상도 설정, 글씨체, 글 크기 설정합니다.
WIN_WIDTH = 600
WIN_HEIGHT = 800
FLOOR = 730
STAT_FONT = pygame.font.SysFont("comicsans", 50)
END_FONT = pygame.font.SysFont("comicsans", 70)
DRAW_LINES = False


WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Flappy Bird")

#게임에 필요한"imgs"폴더 경로에 있는 게임 이미지들을 불러옵니다. 장애물(파이프), 배경화면, 날갯짓 하는 새 그리고 바닥 타일입니다.
pipe_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","pipe.png")).convert_alpha())
bg_img = pygame.transform.scale(pygame.image.load(os.path.join("imgs","bg.png")).convert_alpha(), (600, 900))
bird_images = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird" + str(x) + ".png"))) for x in range(1,4)]
base_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","base.png")).convert_alpha())

#프로그램을 시작할 때  항상 0 세대에서 시작합니다
gen = 0

#객체지향 프로그래밍 입니다

#새(bird) 오브젝트를 나타내는 Class 입니다 
class Bird:
    
    #상황에 따라 변경되는 새 부리의 각도, 프레임 로테이션의 속도, 프레임에서 새의 날갯짓 애니메이션 속도를 설정합니다. x< 느림 x> 빠름
    MAX_ROTATION = 25
    IMGS = bird_images
    ROT_VEL = 20
    ANIMATION_TIME = 5

    #오브젝트 초기화, 새의 시작 위치 (x, y) 파라미터, 새의 기울기 각도, 새의 움직임(물리/physics) 카운트, 새의 시작 속도
    def __init__(self, x, y):
      
        self.x = x
        self.y = y
        self.tilt = 0  
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]
        
        
    #새가 위로 점프하는 함수, 올라가는 속도, 새의 점핑 포인트 추적
    def jump(self):
       
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y

    #새의 움직임 함수, 새가 프레임마다 얼마나 움직였는지 기록
    def move(self):
       
        self.tick_count += 1

        # downward acceleration 새의 현재 속도에 따라 얼마나 위로 또는 아래로 가는지 계산
        displacement = self.vel*(self.tick_count) + 0.5*(3)*(self.tick_count)**2  # calculate displacement

        # terminal velocity 속도가 너무 높거나 너무 낮게 안 나오게 설정
        if displacement >= 16:
            displacement = (displacement/abs(displacement)) * 16

        if displacement < 0:
            displacement -= 2

        self.y = self.y + displacement
        
        #새가 위로 점프하면 새의 부리를 위로 기울기, 떨어지면 새의 부리를 아래로 기울기
        if displacement < 0 or self.y < self.height + 50: 
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:  
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL
    
    #프로그램 윈도우에 새 그리기. 윈도우 파라메터: pygame window 또는 surface
    def draw(self, win):
        
        self.img_count += 1

       #새의 날갯짓 애니메이션, 3개의 이미지가 루프 됨. 애니메이션 카운트에 따라서 새의 이미지가 변경됨
        if self.img_count <= self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count <= self.ANIMATION_TIME*2:
            self.img = self.IMGS[1]
        elif self.img_count <= self.ANIMATION_TIME*3:
            self.img = self.IMGS[2]
        elif self.img_count <= self.ANIMATION_TIME*4:
            self.img = self.IMGS[1]
        elif self.img_count == self.ANIMATION_TIME*4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0

        #새가 아래로 90도 기울어져있으면 날갯짓을 하지 않는다. 다시 점프를 한다면 프레임을 스킵 하지 않고 자연스러운 애니메이션을 보여준다
        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME*2


        # 새의 고정된 이미지를 회전시킬 수 있는 pygame 함수다.
        blitRotateCenter(win, self.img, (self.x, self.y), self.tilt)
    
    #충돌 함수. 현재 새 이미지의 mask를 불러온다
    def get_mask(self):
        
        return pygame.mask.from_surface(self.img)

#파이프/장애물 오브젝트를 나타내는 Class 입니다. 파이프 간의 공간과 그리고 파이프가 움직이는 속도를 설정합니다.
class Pipe():
    
    GAP = 200
    VEL = 5

    #파이프의 랜덤 높이 초기화
    def __init__(self, x):
        
        self.x = x
        self.height = 0

       # 위쪽과 아래쪽 파이프의 위치 설정
        self.top = 0
        self.bottom = 0

        self.PIPE_TOP = pygame.transform.flip(pipe_img, False, True)
        self.PIPE_BOTTOM = pipe_img
        
        #새가 파이프를 이미 지났을 때 설정
        self.passed = False

        self.set_height()
    
    #위쪽 파이프의 위치를 랜덤으로 설정하고 파이프의 높이 설정
    def set_height(self):
        
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP
    
    #파이프가 속도에 따라 움직이는 걸 설정
    def move(self):
        
        self.x -= self.VEL

    #파이프를 프로그램 pygame 윈도우/ surface에 그리는 코드입니다
    def draw(self, win):
       
        # 윗 파이프 그리기
        win.blit(self.PIPE_TOP, (self.x, self.top))
        # 아래 파이프 그리기
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    
    #새의 픽셀과 파이프의 필셀끼리 충돌이 일어나면 return 한다. param bird: Bird object 새와 파이프의 mask를 생성한다
    #offset은 이 mask들의 거리를 측정한다
    def collide(self, bird, win):
        
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)
        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))
        
        #mask가 충돌하는지 확인
        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask,top_offset)

        if b_point or t_point:
            return True

        return False

#바닥/타일 오브젝트를 나타내는 Class 입니다 속도 설정
class Base:
    
    VEL = 5
    WIDTH = base_img.get_width()
    IMG = base_img
    
 #바닥을 초기화 시킨다. param y = int
    def __init__(self, y):
        
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH
    
    #바닥이 각 프레임마다 움직인다. 2개의 이미지를 프레임마다 왼쪽으로 이동시켜 움직이는 것처럼 보이게 한다
    def move(self):
        
        self.x1 -= self.VEL
        self.x2 -= self.VEL
        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH
    
    #바닥을 프로그램 윈도우에 그린다 param win: the pygame surface/window
    def draw(self, win):
        
    
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))

#surface와 blit를 윈도우 창에 맞게 회전
def blitRotateCenter(surf, image, topleft, angle):
 
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center = image.get_rect(topleft = topleft).center)

    surf.blit(rotated_image, new_rect.topleft)
    
#윈도우 창에 새, 파이프, 바닥, 점수, 세대, 그리고 제일 가까운 파이프의 인덱스 그리기
def draw_window(win, birds, pipes, base, score, gen, pipe_ind):

    if gen == 0:
        gen = 1
    win.blit(bg_img, (0,0))

    for pipe in pipes:
        pipe.draw(win)
    
    base.draw(win)
    for bird in birds:
         # 새에서 파이프까지 라인을 그린다
        if DRAW_LINES:
            try:
                pygame.draw.line(win, (255,0,0), (bird.x+bird.img.get_width()/2, bird.y + bird.img.get_height()/2), (pipes[pipe_ind].x + pipes[pipe_ind].PIPE_TOP.get_width()/2, pipes[pipe_ind].height), 5)
                pygame.draw.line(win, (255,0,0), (bird.x+bird.img.get_width()/2, bird.y + bird.img.get_height()/2), (pipes[pipe_ind].x + pipes[pipe_ind].PIPE_BOTTOM.get_width()/2, pipes[pipe_ind].bottom), 5)
            except:
                pass
        #새 그리기
        bird.draw(win)

    # 점수
    score_label = STAT_FONT.render("Score: " + str(score),1,(255,255,255))
    win.blit(score_label, (WIN_WIDTH - score_label.get_width() - 15, 10))

    # 세대
    score_label = STAT_FONT.render("Gens: " + str(gen-1),1,(255,255,255))
    win.blit(score_label, (10, 10))

    #살아남은 새의 수 
    score_label = STAT_FONT.render("Alive: " + str(len(birds)),1,(255,255,255))
    win.blit(score_label, (10, 50))

    pygame.display.update()

#현재 새 인구 시뮬레이션 실행 그리고 각 새들의 fitness를 거리에 따라서 주어진다
def eval_genomes(genomes, config):
    
    global WIN, gen
    win = WIN
    gen += 1
    
    #현재 유전정보와, 유전정보와 관련된 신경망과
    #새 오브젝트가 게임할 때 사용하는 네트워크 리스트들을 생성합니다.
    nets = []
    birds = []
    ge = []
    for genome_id, genome in genomes:
        genome.fitness = 0  # # fitness level 0 으로 시작
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(230,350))
        ge.append(genome)

    base = Base(FLOOR)
    pipes = [Pipe(700)]
    score = 0

    clock = pygame.time.Clock()

    run = True
    while run and len(birds) > 0:
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
                break

        pipe_ind = 0
        if len(birds) > 0:
             #신경망 입력값에 첫 번째 또는 두 번째 파이프를 화면에 사용할지 결정
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():  
                pipe_ind = 1                                                                 

        for x, bird in enumerate(birds): # 매 프레임마다 각 새에게 0.1 fitness를 줘서 살아남도록 설정
            ge[x].fitness += 0.1
            bird.move()

            # 새와 위쪽 파이프 그리고 아래쪽 파이프의 현재 위치를 보내고 신경망을 통해서 점프할지 안 할지 결정됨.
            output = nets[birds.index(bird)].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))

            if output[0] > 0.5:  # tanh 활성화 함수를 사용해 출력값이 -1 과 1 사이가 되도록 합니다. 0.5 이상이면 점프합니다
                bird.jump()

        base.move()

        rem = []
        add_pipe = False
        for pipe in pipes:
            pipe.move()
            # 충돌 확인
            for bird in birds:
                if pipe.collide(bird, win):
                    ge[birds.index(bird)].fitness -= 1
                    nets.pop(birds.index(bird))
                    ge.pop(birds.index(bird))
                    birds.pop(birds.index(bird))

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            if not pipe.passed and pipe.x < bird.x:
                pipe.passed = True
                add_pipe = True

        if add_pipe:
            score += 1
            # 파이프를 통과할때마다 더 많은 보상을 주어지게 설정(not required)
            for genome in ge:
                genome.fitness += 5
            pipes.append(Pipe(WIN_WIDTH))

        for r in rem:
            pipes.remove(r)

        for bird in birds:
            if bird.y + bird.img.get_height() - 10 >= FLOOR or bird.y < -50:
                nets.pop(birds.index(bird))
                ge.pop(birds.index(bird))
                birds.pop(birds.index(bird))

        draw_window(WIN, birds, pipes, base, score, gen, pipe_ind)

        # 점수가 너무 커지면 break
            #if score > 20:
            #pickle.dump(nets[0],open("best.pickle", "wb"))
            #break

#NEAT 알고리즘을 실행하여 신경망을 학습시키고 플래피 버드 게임을 하도록 훈련시킨다. param config_file: location of config file
def run(config_file):
  
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    #인구 생성 top-level object for a NEAT run.
    p = neat.Population(config)

    # 터미널 창에 현재 진행 결과를 보여준다
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
   

    # 50세대까지 실행되도록 설정
    winner = p.run(eval_genomes, 50)

    # 결과/통계를 보여준다
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
     # 구성 파일의 경로 확인.
    #현재 실행되고 있는 directory가 있어도 성공적으로 script 가 실행된다.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)
