using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.Data.Sqlite;
using EndoscopyApp.Models;

namespace EndoscopyApp.Services
{
    public class DatabaseService
    {
        private readonly string _connectionString;

        public DatabaseService()
        {
            var dbPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "endoscopy.db");
            _connectionString = $"Data Source={dbPath}";
            InitializeDatabase();
        }

        private void InitializeDatabase()
        {
            using var connection = new SqliteConnection(_connectionString);
            connection.Open();

            var command = connection.CreateCommand();
            command.CommandText = @"
                CREATE TABLE IF NOT EXISTS Patients (
                    Id INTEGER PRIMARY KEY AUTOINCREMENT,
                    Name TEXT NOT NULL,
                    Age INTEGER,
                    Gender TEXT,
                    Phone TEXT,
                    CreatedAt TEXT
                );

                CREATE TABLE IF NOT EXISTS MediaFiles (
                    Id INTEGER PRIMARY KEY AUTOINCREMENT,
                    PatientId INTEGER,
                    FilePath TEXT NOT NULL,
                    FileType TEXT,
                    CreatedAt TEXT,
                    FOREIGN KEY(PatientId) REFERENCES Patients(Id)
                );
            ";
            command.ExecuteNonQuery();

            // Migration: Add Notes column if it doesn't exist
            try
            {
                command.CommandText = "ALTER TABLE Patients ADD COLUMN Notes TEXT;";
                command.ExecuteNonQuery();
            }
            catch (SqliteException ex) when (ex.SqliteErrorCode == 1) // 1 is usually "duplicate column name"
            {
                // Column already exists, ignore
            }
        }

        public void AddPatient(Patient patient)
        {
            using var connection = new SqliteConnection(_connectionString);
            connection.Open();
            var command = connection.CreateCommand();
            command.CommandText = @"
                INSERT INTO Patients (Name, Age, Gender, Phone, Notes, CreatedAt)
                VALUES ($name, $age, $gender, $phone, $notes, $createdAt);
                SELECT last_insert_rowid();
            ";
            command.Parameters.AddWithValue("$name", patient.Name);
            command.Parameters.AddWithValue("$age", patient.Age);
            command.Parameters.AddWithValue("$gender", patient.Gender);
            command.Parameters.AddWithValue("$phone", patient.Phone);
            command.Parameters.AddWithValue("$notes", (object?)patient.Notes ?? DBNull.Value);
            command.Parameters.AddWithValue("$createdAt", patient.CreatedAt.ToString("o"));

            patient.Id = Convert.ToInt32(command.ExecuteScalar());
        }

        public List<Patient> GetAllPatients()
        {
            var patients = new List<Patient>();
            using var connection = new SqliteConnection(_connectionString);
            connection.Open();
            var command = connection.CreateCommand();
            command.CommandText = "SELECT Id, Name, Age, Gender, Phone, Notes, CreatedAt FROM Patients ORDER BY CreatedAt DESC";
            
            using var reader = command.ExecuteReader();
            while (reader.Read())
            {
                patients.Add(new Patient
                {
                    Id = reader.GetInt32(0),
                    Name = reader.GetString(1),
                    Age = reader.IsDBNull(2) ? 0 : reader.GetInt32(2),
                    Gender = reader.IsDBNull(3) ? "" : reader.GetString(3),
                    Phone = reader.IsDBNull(4) ? "" : reader.GetString(4),
                    Notes = reader.IsDBNull(5) ? "" : reader.GetString(5),
                    CreatedAt = DateTime.Parse(reader.GetString(6))
                });
            }
            return patients;
        }

        public void UpdatePatient(Patient patient)
        {
            using var connection = new SqliteConnection(_connectionString);
            connection.Open();
            var command = connection.CreateCommand();
            command.CommandText = @"
                UPDATE Patients 
                SET Name = $name, Age = $age, Gender = $gender, Phone = $phone, Notes = $notes
                WHERE Id = $id;
            ";
            command.Parameters.AddWithValue("$name", patient.Name);
            command.Parameters.AddWithValue("$age", patient.Age);
            command.Parameters.AddWithValue("$gender", patient.Gender);
            command.Parameters.AddWithValue("$phone", patient.Phone);
            command.Parameters.AddWithValue("$notes", (object?)patient.Notes ?? DBNull.Value);
            command.Parameters.AddWithValue("$id", patient.Id);

            command.ExecuteNonQuery();
        }

        public void DeletePatient(int patientId)
        {
            using var connection = new SqliteConnection(_connectionString);
            connection.Open();
            
            // Delete associated media files first
            var deleteMediaCommand = connection.CreateCommand();
            deleteMediaCommand.CommandText = "DELETE FROM MediaFiles WHERE PatientId = $patientId";
            deleteMediaCommand.Parameters.AddWithValue("$patientId", patientId);
            deleteMediaCommand.ExecuteNonQuery();

            var command = connection.CreateCommand();
            command.CommandText = "DELETE FROM Patients WHERE Id = $id";
            command.Parameters.AddWithValue("$id", patientId);
            command.ExecuteNonQuery();
        }

        // Add more methods as needed for MediaFiles
    }
}
