
import matplotlib.pyplot as plt  
%matplotlib inline

def plot_hour2grade(x, y, xline, yline):
    """ x, y의 값들을 그래프로 출력 """
    plt.figure()  
    plt.plot(x, y, 'ob')  
    plt.plot(xline, yline)
    plt.title('Hours vs. Grade')
    plt.xlabel('hours')
    plt.ylabel('grades')
    plt.show()
