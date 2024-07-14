from abc import ABC, abstractmethod
import stat

class BaseModelsClass(ABC):
    """
    Abstract base class for model classes.
    All model classes should inherit from this class and implement the required methods and properties.

    Properties:
    -----------
    DB_SCHEMA : str
        The schema for creating the table in the database.
    INSERT_QUERY : str
        The query for inserting data into the table.

    Methods:
    --------
    convert_db_response(obj):
        Abstract method to convert a database response to an instance of the subclass.
        Must be implemented by subclasses.
    push_on_db():
        Abstract method to push data to the database.
        Must be implemented by subclasses.
    """

    @property
    @abstractmethod
    def DB_SCHEMA(self):
        """Abstract property for the database schema."""
        pass

    @property
    @abstractmethod
    def INSERT_QUERY(self):
        """Abstract property for the insert query."""
        pass

    @staticmethod
    @abstractmethod
    def convert_db_response(obj):
        """
        Abstract method to convert a database response into an instance of the subclass.

        Parameters:
        -----------
        obj : any
            The database response to be converted.

        Raises:
        -------
        NotImplementedError
            If the method is not implemented by the subclass.
        """
        pass

    @abstractmethod
    def push_on_db(self, notes = 'no notes'):
        """
        Abstract method to push data to the database.

        Raises:
        -------
        NotImplementedError
            If the method is not implemented by the subclass.
        """
        pass
