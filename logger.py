import logging

def setup_logger():
    """
    Setup logger for the AI bot.
    """
    logging.basicConfig(filename="ai_bot.log", level=logging.INFO, 
                        format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger()

logger = setup_logger()
