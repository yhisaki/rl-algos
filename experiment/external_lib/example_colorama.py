import colorama
from colorama import Fore, Back, Style


if __name__ == "__main__":
  # Initializes Colorama
  colorama.init(autoreset=True)

  print(Style.BRIGHT + Back.YELLOW + Fore.RED + "CHEESY")
