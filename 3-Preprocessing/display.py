import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import sys
def main():
	img = sys.argv[1]
	img = mpimg.imread(img)
	plt.imshow(img)
	plt.show()
if __name__=='__main__':
	main()
