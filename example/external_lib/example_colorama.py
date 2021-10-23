import colorama
from colorama import Back, Fore, Style

if __name__ == "__main__":
    # Initializes Colorama
    colorama.init(autoreset=True)

    print(Style.BRIGHT + Back.YELLOW + Fore.RED + "CHEESY")
