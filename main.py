import argparse
from input_output  import valid_txt, read_instance
from instance import Instance

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("instance", help="txt file of the instance to be used.", type=valid_txt)
    args = parser.parse_args()

    filename = args.instance
    print(filename)
    print("running optimization...")
    instance = read_instance(filename)
    #instance._print()

if __name__ == "__main__":
    main()
