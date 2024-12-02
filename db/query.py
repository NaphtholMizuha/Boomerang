import duckdb
import sys

def main():
    table_name = sys.argv[1]
    with duckdb.connect("db/result.db") as con:
        # con.sql(f'select max(avg_acc) from {table_name}').show()
        con.table(table_name).show(max_rows=10086)
    
if __name__ == '__main__':
    main()