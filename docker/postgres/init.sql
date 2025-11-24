-- 创建数据库
CREATE DATABASE codeinsight;

-- 连接到数据库
\c codeinsight;

-- 创建用户表
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    hashed_password VARCHAR(100) NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT true,
    is_superuser BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建项目表
CREATE TABLE projects (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    owner_id INTEGER REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建代码分析结果表
CREATE TABLE code_analysis (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id),
    analysis_type VARCHAR(50) NOT NULL,
    result JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX idx_projects_owner_id ON projects(owner_id);
CREATE INDEX idx_code_analysis_project_id ON code_analysis(project_id);

-- 数据与模型版本管理
CREATE TABLE datasets (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE data_versions (
    id SERIAL PRIMARY KEY,
    dataset_id INTEGER REFERENCES datasets(id) ON DELETE CASCADE,
    version_tag VARCHAR(100) NOT NULL,
    source_hash VARCHAR(128),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(dataset_id, version_tag)
);

CREATE TABLE model_versions (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    version_tag VARCHAR(100) NOT NULL,
    params JSONB,
    metrics JSONB,
    deployed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name, version_tag)
);

CREATE TABLE inference_logs (
    id SERIAL PRIMARY KEY,
    request_id VARCHAR(64),
    endpoint VARCHAR(64),
    model_version_id INTEGER REFERENCES model_versions(id) ON DELETE SET NULL,
    ab_group VARCHAR(8),
    latency_ms DOUBLE PRECISION,
    success BOOLEAN,
    payload_size INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE ab_tests (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    version_a_id INTEGER REFERENCES model_versions(id) ON DELETE CASCADE,
    version_b_id INTEGER REFERENCES model_versions(id) ON DELETE CASCADE,
    ratio_a DOUBLE PRECISION DEFAULT 0.5,
    ratio_b DOUBLE PRECISION DEFAULT 0.5,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP WITH TIME ZONE
);

CREATE TABLE data_quality_runs (
    id SERIAL PRIMARY KEY,
    dataset_version_id INTEGER REFERENCES data_versions(id) ON DELETE CASCADE,
    status VARCHAR(32) NOT NULL,
    metrics JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 索引加速查询
CREATE INDEX idx_data_versions_dataset_id ON data_versions(dataset_id);
CREATE INDEX idx_model_versions_name ON model_versions(name);
CREATE INDEX idx_inference_logs_model_version_id ON inference_logs(model_version_id);