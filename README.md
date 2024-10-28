# Visualizing Partitioning Methods

This project visualizes various partitioning methods using Streamlit. Follow the steps below to set up the project, create a virtual environment, install the required dependencies, and run the app.

## Prerequisites

- Python 3.x installed on your machine
- pip (Python package installer)

## Setup Instructions

### 1. Clone the Repository

First, clone the repository to your local machine:

```sh
git clone https://github.com/Kin-ata/Visualizing-Partitioning-Methods
cd Visualizing-Partitioning-Methods
```

### 2. Create a Virtual Environment

Create a virtual environment to manage the project's dependencies:

```sh
python -m venv venv
```

Activate the virtual environment:

- On Windows:

  ```sh
  .\venv\Scripts\activate
  ```

- On macOS and Linux:

  ```sh
  source venv/bin/activate
  ```

### 3. Install Dependencies

Install the required dependencies listed in `requirements.txt`:

```sh
pip install -r requirements.txt
```

### 4. Run the Streamlit App

Run the Streamlit app using the following command:

```sh
streamlit run app.py
```

This will start the Streamlit server and open the app in your default web browser.

## Project Structure

- `app.py`: Main application file for the Streamlit app.
- `kmeans.py`: K-means clustering implementation.
- `hierarchical.py`: Hierarchical clustering implementation.
- `dbscan.py`: DBSCAN clustering implementation.
- `partitioning.py`: Partitioning clustering implementation.
- `templates.py`: Template configurations for the app.
- `requirements.txt`: List of required Python packages.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [Plotly](https://plotly.com/)
- [scikit-learn](https://scikit-learn.org/)