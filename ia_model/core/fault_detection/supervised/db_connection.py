import pyodbc

def get_db_connection():
    conn_str = (
        "Driver={SQL Server};"
        "Server=OUMAIMA;"
        "Database=Maximo_Local;"  # Replace with your Maximo database name
        "Trusted_Connection=yes;"
    )
    return pyodbc.connect(conn_str)

def get_fault_data():
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # Query to get workorder and fault data
        query = """
        SELECT 
            w.WONUM,
            w.DESCRIPTION,
            w.STATUS,
            w.LOCATION,
            w.ASSETNUM,
            w.WOPRIORITY,
            f.FAILURECODE,
            f.PROBLEMCODE,
            f.FAILUREDESC
        FROM WORKORDER w
        LEFT JOIN FAILUREREPORT f ON w.WONUM = f.WONUM
        WHERE w.STATUS = 'CLOSE'
        """
        
        df = pd.read_sql(query, conn)
        return df
        
    finally:
        conn.close()