import logging
import datetime
import platform
import sys
import shutil
from colorama import init, Fore, Style

# Initialize colorama for cross-platform compatibility
init(autoreset=True)

class BenLOCConfig:
    """Configuration for the BenLOC library."""
    VERSION = "1.0.0"
    TEAM_NAME = "SUFE & Cardinal Optimizer Team"
    CONTACT_EMAIL = "ishongpeili@gmail.com"
    CORE_FEATURES = [
        "Dataset integration for MIP problems",
        "A framework of Learning-based optimizer configuration",
        "Support custom ML models and hyperparameters",
        "Integration with modern solvers (COPT, Gurobi)",
        "Contain several features extracted from COPT's private log"
    ]

class BenLOCLogger:
    """Logger class for BenLOC framework."""

    # Initialize logging configuration
    @staticmethod
    def initialize_logging():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    @staticmethod
    def print_frame():
        """Print a frame line for terminal formatting."""
        terminal_width = shutil.get_terminal_size((80, 20)).columns
        frame_line = "-" * terminal_width
        print(f"{Fore.GREEN}{frame_line}{Style.RESET_ALL}")

    @staticmethod
    def current_datetime():
        """Returns the current date and time."""
        now = datetime.datetime.now()
        return now.strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def system_info():
        """Collects and returns system information."""
        os_name = platform.system()
        python_version = sys.version.split(" ")[0]
        cuda_available = shutil.which("nvidia-smi") is not None
        return os_name, python_version, cuda_available

    @staticmethod
    def print_logo():
        """Print the BenLOC logo."""
        logo = f"""
{Fore.CYAN}                                                        ,--,                             
                                        ,---.'|       ,----..                
            ,---,.                      |   | :      /   /   \\    ,----..   
        ,'  .'  \\                     :   : |     /   .     :  /   /   \\  
        ,---.' .' |               ,---, |   ' :    .   /   ;.  \\|   :     : 
        |   |  |: |           ,-+-. /  |;   ; '   .   ;   /  ` ;.   |  ;. / 
        :   :  :  /   ,---.  ,--.'|'   |'   | |__ ;   |  ; \\ ; |.   ; /--`  
        :   |    ;   /     \\|   |  ,"' ||   | :.'||   :  | ; | ';   | ;     
        |   :     \\ /    /  |   | /  | |'   :    ;.   |  ' ' ' :|   : |     
        |   |   . |.    ' / |   | |  | ||   |  ./ '   ;  \\; /  |.   | '___  
        '   :  '; |'   ;   /|   | |  |/ ;   : ;    \\   \\  ',  / '   ; : .'| 
        |   |  | ; '   |  / |   | |--'  |   ,/      ;   :    /  '   | '/  : 
        |   :   /  |   :    |   |/      '---'        \\   \\ .'   |   :    /  
        |   | ,'    \\   \\  /'---'                     `---`      \\   \\ .'   
        `----'       `----'                                       `---`     
{Style.RESET_ALL}"""
        print(logo)
        print(f"{Fore.YELLOW}Welcome to BenLOC: A Benchmark for Learning-based MIP Optimizer Configuration{Style.RESET_ALL}")
        print(f"This library is developed by {BenLOCConfig.TEAM_NAME}.")
        print(f"Contact us at: {BenLOCConfig.CONTACT_EMAIL}.\n")

    @staticmethod
    def print_dynamic_info():
        """Print system and runtime dynamic information."""
        BenLOCLogger.print_frame()
        logging.info("Collecting system information...")
        print(f"Date & Time: {BenLOCLogger.current_datetime()}")
        os_name, python_version, cuda_available = BenLOCLogger.system_info()
        print(f"Operating System: {os_name}")
        print(f"Python Version: {python_version}")
        print(f"CUDA Available: {'Yes' if cuda_available else 'No'}")
        print(f"Version: BenLOC v{BenLOCConfig.VERSION}")
        print("Core Features:")
        for feature in BenLOCConfig.CORE_FEATURES:
            print(f"  - {feature}")
        BenLOCLogger.print_frame()

    @staticmethod
    def print_guide():
        """Print a user guide for the library."""
        print(f"{Fore.GREEN}Quick Start Guide:{Style.RESET_ALL}")
        print(" See README.md for detailed instructions.")
        # print("  - Run `python main.py` to start.")
        # print("  - Use `python main.py --help` for command-line options.")
        # print("  - Documentation: <link to documentation>")
        print(f"{Fore.YELLOW}Tip: Ensure CUDA is installed for optimal performance.{Style.RESET_ALL}")
        BenLOCLogger.print_frame()


def log_init(func):
    """Decorator to log and print welcome info before executing the main logic."""
    def wrapper(*args, **kwargs):
        BenLOCLogger.print_logo()
        BenLOCLogger.print_dynamic_info()
        BenLOCLogger.print_guide()
        return func(*args, **kwargs)
    return wrapper

if __name__ == "__main__":
    @log_init
    def main():
        pass
    main()