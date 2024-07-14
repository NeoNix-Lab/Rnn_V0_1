# Streamlit Decision Agent Training Application

This Streamlit application allows users to create sets of elements, train decision agents, save training results, and test trained models. It uses SQLite databases for managing data and includes functionalities for adding training configurations, building items, and managing training processes.

## Features

- Create and manage SQLite databases with custom elements.
- Add and manage training configurations.
- Train decision agents and save the results.
- Test trained models to evaluate their performance.

## Installation

### Prerequisites

Ensure you have [Python](https://www.python.org/downloads/) installed on your system (version 3.8 or higher).

### Clone the Repository

Clone this repository to your local machine using:
```sh
git clone [https://github.com/your-repository.git](https://github.com/DevPloyOrg/Rnn_V0_1.git)
cd your-repository
Create a Virtual Environment
Using Conda
Create a Conda environment using the provided conda_environment.yml file:

sh
Copia codice
conda env create -f conda_environment.yml
conda activate myenv
Using pip
Alternatively, you can use pip to install the dependencies:

sh
Copia codice
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
Running the Application
After installing the dependencies, you can run the Streamlit application using:

sh
Copia codice
streamlit run Ui_Exe.py
File Structure
Ui_Add_Training.py: Manages the addition of training configurations and processes.
Ui_Build_Items.py: Manages the creation and building of items.
Ui_Tests.py: Provides configuration settings for database management.
Ui_Training.py: Manages the training processes and iterations.
Ui_Exe.py: Main executable file to run the Streamlit application.
Usage
1. Database Setup
Navigate to the Database Settings page (Ui_Tests.py) to configure and create SQLite databases for managing elements.
You can set paths for data, models, and logs in the Streamlit interface provided.
2. Building Items
Use the Build Items page (Ui_Build_Items.py) to create and manage items within the database.
This includes defining the structure and properties of items that will be used in training.
3. Adding Training Configurations
On the Add Training page (Ui_Add_Training.py), you can add and manage training configurations.
Specify various parameters and settings required for training the decision agents.
4. Training Decision Agents
The Training page (Ui_Training.py) allows you to execute training processes for decision agents.
You can initiate training runs, monitor progress, and save results.
5. Testing Trained Models
Once training is complete, use the Testing page to evaluate the performance of trained models.
This includes loading saved models, running tests, and displaying results.

## Roadmap

We have exciting plans to enhance and expand the capabilities of our application. Below are some of the key features and improvements we are working on:

1. **Improvement of Training Review Section**:
   - Enhance the UI/UX for better visualization and analysis of training sessions.
   - Add more detailed metrics and performance charts.

2. **Improvement of Model Testing Section**:
   - Expand the range of test scenarios and validation techniques.
   - Include more comprehensive reporting and error analysis tools.

3. **Direct Integration with Google Colab**:
   - Allow users to run and train models directly in Google Colab.
   - Provide seamless synchronization of data and models between the local environment and Colab.

4. **Direct Integration with Third-Party Trading Applications**:
   - Integrate with popular trading platforms for live trading and backtesting.
   - Enable automated trading strategies based on trained models.

5. **Multi-Platform Release with MAUI**:
   - Develop and release a multi-platform application using .NET MAUI.
   - Ensure compatibility with Windows, macOS, iOS, and Android.

Stay tuned for updates as we continue to enhance our application and bring new features to our users!


License
This project is licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License - see the LICENSE file for details.

Contact
For any queries or issues, please contact nicoladotmontanari@yahoo.com.
