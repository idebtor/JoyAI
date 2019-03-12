# Artificial Intelligence Applications for ALL

Lecture Notes by idebtor@gmail.com
-------------------
# Getting Started
  __To get started__, do the first thing first:

  1. Read README.
  2. Read Syllabus.
  3. Read '01GettingStarted' - this file
  4. Follow instructions in '01GettingStarted' as soon as possible(ASAP).

  This is available at [github.com/idebtor/JoyAI](https://github.com/idebtor/JoyAI).

## Join Piazza.

There are two ways to join Piazza, go the www.piazza.com.
  - To join Piazza, you may need the following information and
    - School: __Handong Global University__
    - Course: __Artificial Intelligence Applications for ALL__
  - If you have an email address that ends with __~.hgu.edu__ or __~.handong.edu__ domain and use it everyday, go the www.piazza.com and follow the instructions in the website.

  - On your request with your email address, I may register it for you.  We'll be conducting all class-related discussion here this term. The quicker you begin asking questions on Piazza (rather than via emails), the quicker you'll benefit from the collective knowledge of your classmates and instructors. We encourage you to ask questions when you're struggling to understand a concept—you can even do so anonymously.

## Install Anaconda
Anaconda is  a Python and R distribution package. It aims to provide everything you need (python wise) for data science "out of the box".  It includes:
-	The core python language
-	200+ python "packages" (libraries)
-	Spyder (IDE/editor) and Jupyter Notebook
-	conda, Anaconda's own package manager, used for updating Anaconda and packages

#### To install the anaconda:

  - Visit website [Anaconda Distribution](https://www.anaconda.com/distribution/)
    - Choose one of Windows/MacOS/Linux
    - Python 3.x Version Download
  - Need help? Follow [this guide](https://m.blog.naver.com/PostView.nhn?blogId=jooostory&logNo=221196479998&proxyReferer=https%3A%2F%2Fwww.google.com%2F).
  - At the beginning of installation, check the following option
      - Add Anaconda to my PATH environment variable    

#### After your installation
Do the following in cmd windows or in PowerShell to check your successful installation; ($ is just a prompt of your console, >>> is a prompt from Python.)

      ```
      $ python
      >>> import tensorflow as tf
      >>> print(tf.__version__)
      1.13.1
      >>> import keras
      Using TensorFlow backend
      ```

#### Need more installation?
Use the following command if you need more installation of packages (-U for upgrade only):
  ```
  $ pip install a_package_name
  $ pip install -U a_package_name              
  ```

## Install GitHub Desktop
After installation of __GitHub Desktop__, be a member if already not.

  - Clone the GitHub `JoyAI` repository into your local computer:
    - https://github.com/idebtor/JoyAI  

  - How to clone a repository from GitHub:

      - Refer to [this site](https://help.github.com/desktop/guides/contributing-to-projects/cloning-a-repository-from-github-desktop/).
  - Click __'watch'__ and __'star'__ at the top of the web page^^.

  - Then, in your computer, you may have the following github/JoyAI folder as shown below (`user` may be different in your system.):
    - ```C:\Users\user\Documents\GitHub\JoyAI```

  - Since this `JoyAI` repository can be updated anytime, keep this local repository as "read-only".  Don't code yours here!.
  - Copy them into your own repository or your own local development folders in your computer you can easily access them.  They should look like the following:
    ```
    ~/JoyAI/dataset
    ~/JoyAI/deep_class
    ~/JoyAI/deep_code
    ~/JoyAI/ppts
    ~/JoyAI/01GettingStarted.md
    ~/JoyAI/README.md             
    ~/JoyAI/Syllabus       
    ~/JoyAI/Weekly01-02OpenLectures.md

    ```

      __Note for Multi-screen users:__ Remove the following file if GitHub Desktop is displayed off-screen. Restart Desktop GitHub. (`user` below may be different in your system.)
      ```
      C:\Users\user\AppData\Roaming\GitHub Desktop\window-state.json
      ```

## Are ready for 'Hello World!' program in Python?
  - Open a console. (You may use cmd or powershell in Windows.)  

  ```
  $ python
  >>> print('Hello World!')
  >>> 1 + 2
  3
  >>> exit()
  $
  ```

## A few ways to start Jupyter notebook

__Method 1__:
  1. Using Anaconda Navigator, choose Jupyter Notebook.

__Method 2__:

(This option may not work unless you have set PATH environment variable.)
  1. Using File Explorer, navigate to where your Jupyter notebook file is
  2. Using File menu in File Explorer, click Open PowerShell(PS).
  3. At PS console, enter the following:
  ```
  PS C:\> jupyter notebook
  ```

__Method 3__:
This is one-line batch command file that runs Jupyter notebook.
  1. Get a copy of the batch file `start_ipynb.bat` which is available at https://github.com/idebtor/JoyAI
  2. Place the batch file at the folder where your notebook file is. 
  3. Double-click the batch file.
  
__Method 4__: (unstable)

Make an association .ipynb extension with Jupyter Notebook.
  1. `python -m pip install nbopen`
  2. To integrate with your file manager, so you can double click on notebooks to open them, run:

    - Linux/BSD: python -m nbopen.install_xdg

    - Windows: python -m nbopen.install_win

    - Mac: Clone the [repository](https://github.com/takluyver/nbopen) and run ./osx-install.sh



-----------------------------------------------

## Are ready for cloning another open-source program in GitHub?

Let's clone some open-source packages that uses with MNIST dataset and classifies user's hand-writing digits interactively.

  - Clone the following two github sites

  ```
  https://github.com/scrambledpie/Drawing-Mnist-and-Cifarizing-image-files
  https://github.com/rhammell/mnist-draw
  ```

### Drawing-Mnist-and-Cifarizing-image-files
Once you clone this source code, you may see the folder named `Drawing-Mnist-and-Cifarizing-image-files` and ipynb files in your local folder.

The following Jupyter notebook code provides a blank canvas in the notebook so that a user may create their own test samples to input into a trained neural net in Keras. Some code borrowed from Francois Chollet `https://github.com/fchollet`

  - Run the code cells one by one in `DrawMyOwnNumbers.ipynb`
  - If you see some error messages, read and attempt to resolve the problems.
  - You may be asked to install more packages/modules. Then how would you do?

### Minist-draw (unstable) 
  - (I am still experiencing a difficulty to run this program in some machines.)

  - Follow the instructions in README.
    Then you may be able to start a web server and display a web page that has a user interface getting user's hand-writing digits and recognize them interactively.

  - If you experience an error something like

  ```
  127.0.0.1 - - [04/Mar/2019 14:55:50] b'ModuleNotFoundError: No module named \'numpy.core._multiarray_umath\'\r\nImportError: numpy.core.multiarray failed to import\r\n\r\nThe above exception was the direct cause of the following exception:\r\n\r\nTraceback (most recent call last):\r\n  File "<frozen importlib._bootstrap>", line 980,
  ```
  - install numpy with -U (upgrade) option as shown below:
    ```
    pip install -U numpy
    ```

## References
- [DrawMyOwnNumbers Notebook](https://github.com/scrambledpie/Drawing-Mnist-and-Cifarizing-image-files)
- [mnist-draw](https://github.com/rhammell/mnist-draw)

## What's Next?

- To learning Python and Numpy from the beginning, visit [this site](https://www.learnpython.org/en/Welcome). It is offered by learnpython.org.
-

----------------------------
_One thing I know, I was blind but now I see. John 9:25_

----------------------------
