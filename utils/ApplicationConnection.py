import psycopg2
from psycopg2.extras import LogicalReplicationConnection
from threading import Lock
from Common.constants import *

class ApplicationConnection:
    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super(ApplicationConnection, cls).__new__(cls)
        return cls._instance

    def __init__(self, dbname, host, user, password):
        if not hasattr(self, "_initialized"):
            self.dbname = DB_NAME
            self.host = DB_HOST
            self.user = DB_USER
            self.password = DB_PASS
            self._connection = None
            self._initialized = True

    def connect(self, replication=False):

        if self._connection is None or self._connection.closed:
            try:
                if replication:
                    self._connection = psycopg2.connect(
                        f"dbname='{self.dbname}' host='{self.host}' user='{self.user}' password='{self.password}'",
                        connection_factory=LogicalReplicationConnection
                    )
                else:
                    self._connection = psycopg2.connect(
                        f"dbname='{self.dbname}' host='{self.host}' user='{self.user}' password='{self.password}'"
                    )
            except Exception as e:
                raise Exception(f"Error connecting to the database: {e}")
        return self._connection