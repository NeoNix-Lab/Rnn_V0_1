import json
import os

#TODO: Menage db configuration for colab

class Config:
    """
    Class to manage the application's configuration.

    This class handles loading and saving configuration settings from a JSON file.
    It manages the paths to databases and log directories based on the environment.

    Attributes:
    -----------
    CONFIG_FILE : str
        The path to the configuration file.
    STREAMLIT_ENVIROMENT : str
        Constant representing the Streamlit environment.
    COLAB_ENVIROMENT : str
        Constant representing the Colab environment.

    Methods:
    --------
    load_config()
        Loads the configuration from the JSON file.
    save_config()
        Saves the configuration to the JSON file.
    data_path()
        Getter for the data database path based on the environment.
    data_path(value)
        Setter for the data database path based on the environment.
    models_path()
        Getter for the models database path based on the environment.
    models_path(value)
        Setter for the models database path based on the environment.
    logs_path()
        Getter for the logs directory path based on the environment.
    logs_path(value)
        Setter for the logs directory path based on the environment.
    """

    CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
    STREAMLIT_ENVIROMENT = 'streamlit'
    COLAB_ENVIROMENT = 'colab'
    
    def __init__(self, enviroment=STREAMLIT_ENVIROMENT):
        self.enviroment = enviroment
        self.config = self.load_config()

    def load_config(self):
        """Loads the configuration from the JSON file."""
        if os.path.exists(self.CONFIG_FILE):
            with open(self.CONFIG_FILE, 'r') as f:
                return json.load(f)
        return {}

    def save_config(self):
        """Saves the configuration to the JSON file."""
        with open(self.CONFIG_FILE, 'w') as f:
            json.dump(self.config, f, indent=4)

    @property
    def data_path(self):
        """Getter for the data database path based on the environment."""
        if self.enviroment == self.STREAMLIT_ENVIROMENT:
            return self.config.get('DATA_PATH', '')
        elif self.enviroment == self.COLAB_ENVIROMENT:
            return self.config.get('DATA_PATH_COLAB', '')

    @data_path.setter
    def data_path(self, value):
        """Setter for the data database path based on the environment."""
        if self.enviroment == self.STREAMLIT_ENVIROMENT:
            self.config['DATA_PATH'] = value
        elif self.enviroment == self.COLAB_ENVIROMENT:
            self.config['DATA_PATH_COLAB'] = value

    @property
    def models_path(self):
        """Getter for the models database path based on the environment."""
        if self.enviroment == self.STREAMLIT_ENVIROMENT:
            return self.config.get('MODLES_PATH', '')
        elif self.enviroment == self.COLAB_ENVIROMENT:
            return self.config.get('MODLES_PATH_COLAB', '')

    @models_path.setter
    def models_path(self, value):
        """Setter for the models database path based on the environment."""
        if self.enviroment == self.STREAMLIT_ENVIROMENT:
            self.config['MODLES_PATH'] = value
        elif self.enviroment == self.COLAB_ENVIROMENT:
            self.config['MODLES_PATH_COLAB'] = value

    @property
    def logs_path(self):
        """Getter for the logs directory path based on the environment."""
        if self.enviroment == self.STREAMLIT_ENVIROMENT:
            return self.config.get('LOGS_PATH', '')
        elif self.enviroment == self.COLAB_ENVIROMENT:
            return self.config.get('LOGS_PATH_COLAB', '')

    @logs_path.setter
    def logs_path(self, value):
        """Setter for the logs directory path based on the environment."""
        if self.enviroment == self.STREAMLIT_ENVIROMENT:
            self.config['LOGS_PATH'] = value
        elif self.enviroment == self.COLAB_ENVIROMENT:
            self.config['LOGS_PATH_COLAB'] = value
