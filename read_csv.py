import pandas as pd
import sys

def main():
	csv_file = sys.argv[1]
	csv_ = pd.read_csv(csv_file)
	print(csv_)

if __name__=='__main__':
	main()
