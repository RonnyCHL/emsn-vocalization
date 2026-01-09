-- ===========================================
-- Run dit als postgres user op de NAS
-- ssh naar NAS, dan:
-- docker exec -i postgres psql -U postgres -d emsn < create_tables.sql
-- ===========================================

-- Tabel voor Xeno-canto audio metadata (locaties)
CREATE TABLE IF NOT EXISTS xeno_canto_recordings (
    id SERIAL PRIMARY KEY,
    species_name VARCHAR(100) NOT NULL,
    xc_id VARCHAR(20),
    country VARCHAR(100),
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    vocalization_type VARCHAR(20),
    quality CHAR(1),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabel voor confusion matrix data
CREATE TABLE IF NOT EXISTS vocalization_confusion_matrix (
    id SERIAL PRIMARY KEY,
    species_name VARCHAR(100) NOT NULL,
    true_label VARCHAR(20) NOT NULL,
    predicted_label VARCHAR(20) NOT NULL,
    count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(species_name, true_label, predicted_label)
);

-- Index voor snelle queries
CREATE INDEX IF NOT EXISTS idx_xc_species ON xeno_canto_recordings(species_name);
CREATE INDEX IF NOT EXISTS idx_xc_country ON xeno_canto_recordings(country);
CREATE INDEX IF NOT EXISTS idx_cm_species ON vocalization_confusion_matrix(species_name);

-- Grant rechten aan birdpi_zolder
GRANT SELECT, INSERT, UPDATE, DELETE ON xeno_canto_recordings TO birdpi_zolder;
GRANT SELECT, INSERT, UPDATE, DELETE ON vocalization_confusion_matrix TO birdpi_zolder;
GRANT USAGE, SELECT ON SEQUENCE xeno_canto_recordings_id_seq TO birdpi_zolder;
GRANT USAGE, SELECT ON SEQUENCE vocalization_confusion_matrix_id_seq TO birdpi_zolder;

-- Bevestiging
SELECT 'Tables created successfully!' as status;
