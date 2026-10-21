-- CS619 — Student Performance Predictor
-- Phase 4 — MySQL 8.0 Database Setup Script
-- Authors: Syeda Farkhanda Batool Naqvi (BC220424824)
--          Saad Shakir (BC220404989)


CREATE DATABASE IF NOT EXISTS cs619_studentdb
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE cs619_studentdb;

-- Students Table 
CREATE TABLE IF NOT EXISTS students (
    id                    INT AUTO_INCREMENT PRIMARY KEY,
    Student_ID            VARCHAR(20)  UNIQUE NOT NULL,
    Registration_No       VARCHAR(30)  DEFAULT '',
    Gender                VARCHAR(10)  DEFAULT '',
    Age                   INT          DEFAULT 0,
    Course                VARCHAR(60)  DEFAULT '',
    Semester              INT          DEFAULT 0,
    Section               VARCHAR(10)  DEFAULT '',
    Total_Classes         INT          DEFAULT 0,
    Classes_Attended      INT          DEFAULT 0,
    Attendance_Percentage FLOAT        DEFAULT 0,
    Assignment_1          FLOAT        DEFAULT 0,
    Assignment_2          FLOAT        DEFAULT 0,
    Assignment_3          FLOAT        DEFAULT 0,
    Average_Assignment    FLOAT        DEFAULT 0,
    Quiz_1                FLOAT        DEFAULT 0,
    Quiz_2                FLOAT        DEFAULT 0,
    Average_Quiz          FLOAT        DEFAULT 0,
    Midterm_Exam          FLOAT        DEFAULT 0,
    Final_Exam            FLOAT        DEFAULT 0,
    Lab_Work_1            FLOAT        DEFAULT 0,
    Lab_Work_2            FLOAT        DEFAULT 0,
    Average_Lab           FLOAT        DEFAULT 0,
    Project_Score         FLOAT        DEFAULT 0,
    CGPA                  FLOAT        DEFAULT 0,
    Previous_GPA          FLOAT        DEFAULT 0,
    Study_Hours_Per_Week  FLOAT        DEFAULT 0,
    Library_Visits        INT          DEFAULT 0,
    Online_Course_Hours   FLOAT        DEFAULT 0,
    Class_Participation   FLOAT        DEFAULT 0,
    Group_Work_Score      FLOAT        DEFAULT 0,
    Sports_Activity       VARCHAR(10)  DEFAULT '',
    Club_Membership       VARCHAR(5)   DEFAULT '',
    Total_Score           FLOAT        DEFAULT 0,
    Performance           VARCHAR(20)  DEFAULT ''
);

-- Users Table (UNIQUE on username prevents duplicate teacher/student rows)
CREATE TABLE IF NOT EXISTS users (
    id         INT AUTO_INCREMENT PRIMARY KEY,
    username   VARCHAR(50)  NOT NULL UNIQUE,
    password   VARCHAR(100) NOT NULL,
    role       VARCHAR(20)  NOT NULL,
    student_id VARCHAR(20)  DEFAULT NULL
);

-- Default teacher login
INSERT IGNORE INTO users (username, password, role, student_id)
VALUES ('teacher', 'teacher123', 'teacher', NULL);

--  Predictions Table 
CREATE TABLE IF NOT EXISTS predictions (
    id               INT AUTO_INCREMENT PRIMARY KEY,
    Student_ID       VARCHAR(20) UNIQUE NOT NULL,
    Predicted_Result VARCHAR(10) DEFAULT '',
    Fail_Probability FLOAT       DEFAULT 0,
    Risk_Level       VARCHAR(20) DEFAULT '',
    created_at       TIMESTAMP   DEFAULT CURRENT_TIMESTAMP
                                 ON UPDATE CURRENT_TIMESTAMP
);

--  Confirmation 
SELECT 'Database setup complete! Now run: python app.py' AS Status;
SELECT table_name FROM information_schema.tables WHERE table_schema = 'cs619_studentdb';