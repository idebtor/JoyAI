from math import *
import random
import copy


class TicTacToe:
    def __init__(self, state = [0,0,0,0,0,0,0,0,0]):
        self.playerJustMoved = 2
        self.state = state

    def Clone(self): # TicTacToe에 대한 상황 카피.
        state = TicTacToe()
        state.state = self.state[:]
        state.playerJustMoved = self.playerJustMoved
        return state

    def DoMove(self, move): #index범위 안에 있고 빈칸일 경우, 실행. (1 아니면 2 입력)
        assert move >= 0 and move <= 8 and self.state[move] == 0
        self.playerJustMoved = 3 - self.playerJustMoved
        self.state[move] = self.playerJustMoved


    def GetMoves(self):
        if self.checkState() != 0: #게임 결과가 결정된 경우
            return [] #빈 list return

        else: #게임이 진행된 경우
            moves = [] #해당 칸 입력받아서 리턴.
            for i in range(9):
                if self.state[i] == 0: #빈칸 값만 받음.
                    moves.append(i)

            return moves

    def GetResult(self, playerjm):
        result = self.checkState()
        assert result != 0
        if result == -1:
            return 0.5 #무승부일 경우 0.5 return

        elif result == playerjm:
            return 1.0 #승리할 경우 1 return
        else:
            return 0.0 #패배할 경우 0 return


    def checkState(self): #게임 승리 판별
        for (x,y,z) in [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]:
            if self.state[x] == self.state[y] == self.state[z]:
                if self.state[x] == 1:
                    return 1
                elif self.state[x] == 2:
                    return 2

        #
        if [i for i in range(9) if self.state[i] == 0] == []: #0으로 채워진 셀이 없는 경우
            return -1 #무승부 return
        return 0 #나머지 return 0

    #화면 표시.
    def __repr__(self): #시스템이 해당 객체를 이해할 수 있는 형식으로 전환해줌.
        s = ""
        for i in range(9):
            s += ".0X"[self.state[i]]
            if i % 3 == 2:
                s += "\n"
        return s

class Node:
    def __init__(self, move = None, parent = None, state = None):
        self.move = move
        self.parentNode = parent
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.untriedMoves = state.GetMoves()
        self.playerJustMoved = state.playerJustMoved

    def UCTSelectChild(self):
        s = sorted(self.childNodes, key = lambda c: c.wins/c.visits + sqrt(2 * log(self.visits) / c.visits)) #알고리즘을 통한 정렬
        return s[-1]  #마지막값(가장 큰 값) 리턴

    def AddChild(self, m ,s): #부모노드에 자식 노드 추가
        n = Node(move = m, parent = self, state = copy.deepcopy(s))
        self.untriedMoves.remove(m)
        self.childNodes.append(n)
        return n

    def Update(self, result): #게임 결과 입력
        self.visits += 1
        self.wins += result

    def __repr__(self): #훈련 결과값 출력
        return "[M" + str(self.move) + " W/V " + str(self.wins) + "/" + str(self.visits) + " U" + str(self.untriedMoves) + "]"

    def ChildrenToString(self):
        s =""
        for c in self.childNodes:
            s += str(c) + "\n"
        return s


def UCT(rootstate, itermax):
    rootnode = Node(state = rootstate)

    for i in range(itermax):
        node = rootnode #tictactoe
        state = copy.deepcopy(rootstate)

        #selection #적절한 자식노드 선택
        while node.untriedMoves == [] and node.childNodes != []:
            node = node.UCTSelectChild()
            state.DoMove(node.move)

        #Expansion #탐험, 선택되지 않은 자식노드 추가
        if node.untriedMoves != []:
            m = random.choice(node.untriedMoves)
            state.DoMove(m)
            node = node.AddChild(m, state)

        #simulation #선태된 노드에 대해서 랜덤하게 게임 진행
        while state.GetMoves() != []:
            state.DoMove(random.choice(state.GetMoves()))

        #BackPropagation 결과값에따른 노드 업데이트, 승률(가중치) 입력.
        while node != None:
            node.Update(state.GetResult(node.playerJustMoved))
            node = node.parentNode

    print (rootnode.ChildrenToString())
    #승률에 따라서 자식 노드들 정렬
    s = sorted(rootnode.childNodes, key = lambda c: c.wins/c.visits)
    return sorted(s, key = lambda c: c.visits)[-1].move #가장 승률이 높은 값 결정.


def UCTPlayGame():
    state = TicTacToe()
    while state.GetMoves() != []:
        print (str(state))
        if state.playerJustMoved == 2:
            rootstate = copy.deepcopy(state)
            m = UCT(rootstate, itermax = 10000)
        else:
            m = int(input("which Do you want? : "))
            m -= 1
        print ("Best Move : " + str(m+1) + "\n")
        state.DoMove(m)

    if state.GetResult(state.playerJustMoved) == 1.0:
        print ("Player " + str(state.playerJustMoved) + " Wins!!")

    elif state.GetResult(state.playerJustMoved) == 0.0:
        print ("Payer " + str(3 - state.playerJustMoved) + " Wins!!")

    else: print ("Draw!!") # 무승부


if __name__ == "__main__":
    UCTPlayGame() #메인 함수 실행.
