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

            using var command = connection.CreateCommand();

            // Create Patients table
            command.CommandText = @"
                CREATE TABLE IF NOT EXISTS Patients (
                    Id INTEGER PRIMARY KEY AUTOINCREMENT,
                    Nic TEXT,
                    Name TEXT NOT NULL,
                    Age INTEGER,
                    Gender TEXT,
                    Phone TEXT,
                    Notes TEXT,
                    CreatedAt TEXT
                );
            ";
            command.ExecuteNonQuery();

            // Create MediaFiles table
            command.CommandText = @"
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

            // Migration: Check if Nic column exists (for older versions)
            try
            {
                command.CommandText = "ALTER TABLE Patients ADD COLUMN Nic TEXT;";
                command.ExecuteNonQuery();
            }
            catch { /* Ignore if exists */ }

            // Migration: Check if Notes column exists
            try
            {
                command.CommandText = "ALTER TABLE Patients ADD COLUMN Notes TEXT;";
                command.ExecuteNonQuery();
            }
            catch { /* Ignore if exists */ }
        }

        public void AddPatient(Patient patient)
        {
            using var connection = new SqliteConnection(_connectionString);
            connection.Open();

            using var transaction = connection.BeginTransaction();
            try
            {
                var command = connection.CreateCommand();
                command.Transaction = transaction;
                command.CommandText = @"
                    INSERT INTO Patients (Nic, Name, Age, Gender, Phone, Notes, CreatedAt)
                    VALUES ($nic, $name, $age, $gender, $phone, $notes, $createdAt);
                ";
                command.Parameters.AddWithValue("$nic", (object?)patient.Nic ?? DBNull.Value);
                command.Parameters.AddWithValue("$name", patient.Name);
                command.Parameters.AddWithValue("$age", patient.Age);
                command.Parameters.AddWithValue("$gender", patient.Gender);
                command.Parameters.AddWithValue("$phone", patient.Phone);
                command.Parameters.AddWithValue("$notes", (object?)patient.Notes ?? DBNull.Value);
                command.Parameters.AddWithValue("$createdAt", patient.CreatedAt.ToString("o"));

                command.ExecuteNonQuery();

                // Get the generated ID
                command.CommandText = "SELECT last_insert_rowid();";
                patient.Id = Convert.ToInt32(command.ExecuteScalar());

                transaction.Commit();
            }
            catch
            {
                transaction.Rollback();
                throw;
            }
        }

        public List<Patient> GetAllPatients()
        {
            var patients = new List<Patient>();
            using var connection = new SqliteConnection(_connectionString);
            connection.Open();
            var command = connection.CreateCommand();
            command.CommandText = "SELECT Id, Nic, Name, Age, Gender, Phone, Notes, CreatedAt FROM Patients ORDER BY CreatedAt DESC";

            using var reader = command.ExecuteReader();
            while (reader.Read())
            {
                patients.Add(new Patient
                {
                    Id = reader.GetInt32(0),
                    Nic = reader.IsDBNull(1) ? "" : reader.GetString(1),
                    Name = reader.GetString(2),
                    Age = reader.IsDBNull(3) ? 0 : reader.GetInt32(3),
                    Gender = reader.IsDBNull(4) ? "" : reader.GetString(4),
                    Phone = reader.IsDBNull(5) ? "" : reader.GetString(5),
                    Notes = reader.IsDBNull(6) ? "" : reader.GetString(6),
                    CreatedAt = DateTime.Parse(reader.GetString(7))
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
                SET Nic = $nic, Name = $name, Age = $age, Gender = $gender, Phone = $phone, Notes = $notes
                WHERE Id = $id;
            ";
            command.Parameters.AddWithValue("$nic", (object?)patient.Nic ?? DBNull.Value);
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
    }
}

