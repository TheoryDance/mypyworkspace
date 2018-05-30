import pymysql

conn = pymysql.connect("localhost", "root", "123456", "test")
cur = conn.cursor()
# cur.execute("create table student_py(stuId int primary key,stuName varchar(20))");
cur.executemany("insert into student_py(stuId,stuName) values(%s,%s)",
                [
                    (3,'ranfs'),
                    (4,'ranfs'),
                    (5,'ranfs'),
                ]
                )
cur.close()
conn.commit()
conn.close()
