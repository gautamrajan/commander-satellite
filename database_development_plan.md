# Dumpster Database Development Plan

## Current State Analysis

### What We Have
- **Detection Pipeline**: Complete LLM-based scanning system using OpenRouter/Gemini
- **Data Outputs**: JSONL files containing detection results with geographic coordinates
- **Review System**: Flask web app for manual validation of detections
- **File-based Storage**: All data currently stored in JSONL files
- **Spatial Data**: Tile coordinates (z/x/y) that can be converted to lat/lon

### Current Data Schema
```json
// all_results.jsonl - Full scan results
{
  "path": "22/721538/1677304.jpg", 
  "z": 22, "x": 721538, "y": 1677304,
  "model": "google/gemini-2.5-pro",
  "result_raw": {"dumpster": false, "confidence": 0.1},
  "positive": false,
  "confidence": 0.1,
  "error": null
}

// dumpsters.jsonl - Positive detections only
{
  "path": "22/721538/1677276.jpg",
  "z": 22, "x": 721538, "y": 1677276,
  "confidence": 0.6,
  "model": "google/gemini-2.5-pro"
}

// reviewed_results.jsonl - Human validation
{
  "path": "22/721538/1677276.jpg",
  "approved": false
}
```

---

## Vision: Production-Ready Dumpster Database

### Target Architecture
- **Database**: PostgreSQL with PostGIS for geospatial operations
- **API Layer**: FastAPI for RESTful services and real-time data access
- **Enhanced Data Model**: Rich dumpster entities with business context
- **Scalability**: Support for multiple scan areas and historical data
- **Integration Ready**: APIs for CRM integration and business intelligence

---

## Phase 1: Database Foundation (Weeks 1-2)

### Milestone 1.1: Database Setup & Schema Design

**Tasks:**
1. **Set up PostgreSQL with PostGIS**
   - Install PostgreSQL and PostGIS extension
   - Create development database
   - Configure connection pooling

2. **Design Core Schema**
   ```sql
   -- Geographic Areas/Scan Sessions
   CREATE TABLE scan_areas (
     id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
     name VARCHAR(255) NOT NULL,
     description TEXT,
     bounds GEOMETRY(Polygon, 4326),
     created_at TIMESTAMP DEFAULT NOW(),
     metadata JSONB
   );

   -- Individual Tile Scans
   CREATE TABLE tile_scans (
     id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
     scan_area_id UUID REFERENCES scan_areas(id),
     tile_path VARCHAR(255) NOT NULL,
     z INTEGER NOT NULL,
     x INTEGER NOT NULL,
     y INTEGER NOT NULL,
     center_point GEOMETRY(Point, 4326),
     tile_bounds GEOMETRY(Polygon, 4326),
     scanned_at TIMESTAMP DEFAULT NOW(),
     model_used VARCHAR(100),
     scan_metadata JSONB
   );

   -- Dumpster Detections
   CREATE TABLE dumpster_detections (
     id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
     tile_scan_id UUID REFERENCES tile_scans(id),
     location GEOMETRY(Point, 4326),
     confidence DECIMAL(4,3) NOT NULL,
     raw_model_output JSONB,
     human_reviewed BOOLEAN DEFAULT FALSE,
     human_approved BOOLEAN,
     reviewed_at TIMESTAMP,
     reviewed_by VARCHAR(255),
     created_at TIMESTAMP DEFAULT NOW()
   );

   -- Business Context (Future)
   CREATE TABLE business_locations (
     id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
     name VARCHAR(255),
     address TEXT,
     business_type VARCHAR(100),
     location GEOMETRY(Point, 4326),
     estimated_employees INTEGER,
     created_at TIMESTAMP DEFAULT NOW()
   );

   -- Dumpster-Business Relationships
   CREATE TABLE dumpster_business_links (
     dumpster_id UUID REFERENCES dumpster_detections(id),
     business_id UUID REFERENCES business_locations(id),
     confidence DECIMAL(4,3),
     link_type VARCHAR(50), -- 'proximity', 'visual_connection', 'manual'
     created_at TIMESTAMP DEFAULT NOW(),
     PRIMARY KEY (dumpster_id, business_id)
   );
   ```

3. **Create Database Migration System**
   - Set up Alembic for schema versioning
   - Create initial migration scripts

**Deliverables:**
- PostgreSQL database with PostGIS
- Complete schema with indexes and constraints
- Migration system ready for production

### Milestone 1.2: Data Migration Scripts

**Tasks:**
1. **Convert Tile Coordinates to Geographic Coordinates**
   ```python
   def tile_to_latlon(z: int, x: int, y: int) -> Tuple[float, float]:
       """Convert tile coordinates to latitude/longitude"""
       n = 2.0 ** z
       lon_deg = x / n * 360.0 - 180.0
       lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
       lat_deg = math.degrees(lat_rad)
       return lat_deg, lon_deg
   ```

2. **JSONL to Database Migration Tool**
   - Parse existing JSONL files
   - Convert tile coordinates to proper geometries
   - Populate database with historical scan data
   - Preserve all metadata and relationships

3. **Data Validation & Quality Checks**
   - Verify coordinate transformations
   - Check for duplicate detections
   - Validate confidence scores and model outputs

**Deliverables:**
- `migrate_existing_data.py` script
- All historical scan data migrated to database
- Data quality validation reports

---

## Phase 2: API Development (Weeks 3-4)

### Milestone 2.1: FastAPI Backend

**Tasks:**
1. **Core API Structure**
   ```python
   # FastAPI app with database models
   from fastapi import FastAPI, Depends
   from sqlalchemy.orm import Session
   from geoalchemy2 import Geography
   
   app = FastAPI(title="Dumpster Detection API")
   
   # Example endpoints:
   # GET /scan-areas - List all scanned areas
   # POST /scan-areas - Create new scan area
   # GET /scan-areas/{id}/detections - Get detections in area
   # GET /detections - Search detections with geographic filters
   # POST /detections/{id}/review - Submit human review
   # GET /analytics/coverage - Get coverage statistics
   ```

2. **Database Integration**
   - SQLAlchemy models with PostGIS integration
   - Connection pooling and query optimization
   - Geographic query support (within radius, bounding box, etc.)

3. **Key Endpoints**
   - **Detection Management**: CRUD operations for detections
   - **Geographic Queries**: Find dumpsters within area/radius
   - **Review Workflow**: Submit and track human reviews
   - **Analytics**: Statistics and coverage metrics
   - **Export**: JSON, CSV, GeoJSON formats

**Deliverables:**
- Complete FastAPI application
- Database integration with geographic queries
- Interactive API documentation (Swagger)

### Milestone 2.2: Enhanced Scanning Integration

**Tasks:**
1. **Modify Scanning Scripts for Database Output**
   - Update `scan_dumpsters.py` to write directly to database
   - Maintain JSONL output for backward compatibility
   - Add scan session management

2. **Real-time Scanning Progress**
   - WebSocket updates for live scan progress
   - Database-backed scan status tracking
   - Better error handling and recovery

3. **Batch Processing Improvements**
   - Queue system for large area processing
   - Resume capability with database state tracking
   - Parallel processing optimization

**Deliverables:**
- Database-integrated scanning pipeline
- Real-time progress tracking
- Improved batch processing capabilities

---

## Phase 3: Enhanced Data Model (Weeks 5-6)

### Milestone 3.1: Dumpster Classification & Sizing

**Tasks:**
1. **Enhanced Detection Schema**
   ```sql
   ALTER TABLE dumpster_detections ADD COLUMN size_category VARCHAR(20); -- small, medium, large
   ALTER TABLE dumpster_detections ADD COLUMN dumpster_type VARCHAR(50); -- front-load, rear-load, roll-off
   ALTER TABLE dumpster_detections ADD COLUMN estimated_volume_yards INTEGER;
   ALTER TABLE dumpster_detections ADD COLUMN service_provider VARCHAR(100);
   ALTER TABLE dumpster_detections ADD COLUMN last_emptied_estimate DATE;
   ```

2. **Size Classification Algorithm**
   - Pixel analysis for dumpster size estimation
   - Reference object comparison (cars, buildings)
   - Machine learning model for size prediction

3. **Service Provider Detection**
   - Logo/branding recognition in imagery
   - Color pattern analysis
   - Integration with known fleet colors

**Deliverables:**
- Enhanced dumpster classification system
- Size estimation algorithms
- Service provider detection capability

### Milestone 3.2: Business Context Integration

**Tasks:**
1. **Business Location Data Integration**
   - API integration with business databases (Google Places, Yelp)
   - Address geocoding and business type classification
   - Proximity-based dumpster-business linking

2. **Waste Generation Modeling**
   ```python
   def estimate_waste_generation(business_type: str, employees: int, 
                                dumpster_size: str) -> Dict[str, float]:
       """Estimate waste generation and optimal service frequency"""
       # Business-specific waste generation models
   ```

3. **Territory and Route Analysis**
   - Service area optimization
   - Route efficiency scoring
   - Competitive landscape mapping

**Deliverables:**
- Business context database
- Waste generation estimation models
- Territory analysis capabilities

---

## Phase 4: Business Intelligence & Analytics (Weeks 7-8)

### Milestone 4.1: Analytics Dashboard API

**Tasks:**
1. **Market Intelligence Endpoints**
   ```python
   # Analytics API endpoints
   GET /analytics/market-coverage/{area_id}
   GET /analytics/competitor-presence/{area_id}
   GET /analytics/opportunity-scoring/{area_id}
   GET /analytics/waste-density-heatmap/{area_id}
   ```

2. **Lead Scoring Algorithm**
   - Prioritize prospects based on dumpster analysis
   - Factor in business size, waste patterns, competition
   - Generate qualified lead lists for sales teams

3. **Competitive Analysis**
   - Map competitor presence by service provider detection
   - Identify underserved areas and market gaps
   - Track market share and penetration rates

**Deliverables:**
- Analytics API with market intelligence
- Lead scoring algorithms
- Competitive analysis tools

### Milestone 4.2: Export & Integration Capabilities

**Tasks:**
1. **Enhanced Export Formats**
   - CRM-ready contact lists with business context
   - Sales territory assignments
   - Route optimization data exports
   - GIS-compatible mapping formats

2. **CRM Integration Preparation**
   - Salesforce API integration framework
   - Lead synchronization workflows
   - Custom field mapping for waste management context

3. **Business Intelligence Connectors**
   - Tableau/Power BI data connectors
   - Automated reporting pipelines
   - KPI tracking and alerting

**Deliverables:**
- Multi-format export capabilities
- CRM integration framework
- BI tool connectors

---

## Phase 5: Production Deployment (Weeks 9-10)

### Milestone 5.1: Production Infrastructure

**Tasks:**
1. **Docker Containerization**
   - Multi-container setup (API, database, workers)
   - Development and production configurations
   - Health checks and monitoring

2. **Cloud Deployment**
   - AWS/GCP infrastructure setup
   - Managed PostgreSQL with PostGIS
   - Load balancing and auto-scaling
   - CDN for tile serving

3. **Security & Compliance**
   - API authentication and authorization
   - Data encryption at rest and in transit
   - Audit logging and compliance features

**Deliverables:**
- Production-ready containerized deployment
- Cloud infrastructure with monitoring
- Security and compliance framework

### Milestone 5.2: Performance Optimization & Monitoring

**Tasks:**
1. **Database Optimization**
   - Query performance tuning
   - Spatial indexing optimization
   - Database connection pooling

2. **API Performance**
   - Caching strategies (Redis)
   - Rate limiting and throttling
   - Geographic query optimization

3. **Monitoring & Alerting**
   - Application performance monitoring
   - Database health monitoring
   - Automated alerting for scan failures

**Deliverables:**
- Optimized production performance
- Comprehensive monitoring system
- Automated alerting and recovery

---

## Technology Stack Summary

### Core Technologies
- **Database**: PostgreSQL 15+ with PostGIS 3.3+
- **API Framework**: FastAPI with Pydantic models
- **Geographic Processing**: PostGIS, Shapely, GeoPandas
- **Database ORM**: SQLAlchemy with GeoAlchemy2
- **Migration System**: Alembic
- **Task Queue**: Celery with Redis
- **Caching**: Redis
- **Containerization**: Docker & Docker Compose

### Development Tools
- **Testing**: pytest with geographic test fixtures
- **API Documentation**: Swagger/OpenAPI (built into FastAPI)
- **Code Quality**: black, flake8, mypy
- **Monitoring**: Prometheus + Grafana
- **Logging**: structured logging with geographic context

### Cloud Infrastructure (Production)
- **Compute**: AWS ECS or Google Cloud Run
- **Database**: AWS RDS PostgreSQL or Google Cloud SQL
- **Storage**: S3 or Google Cloud Storage for tiles
- **CDN**: CloudFront or Cloud CDN for tile serving
- **Monitoring**: AWS CloudWatch or Google Cloud Monitoring

---

## Expected Outcomes

### Technical Outcomes
- **Scalable Database**: Handle millions of detections across multiple cities
- **High-Performance API**: Sub-second response times for geographic queries
- **Production Ready**: Containerized, monitored, and scalable deployment
- **Integration Ready**: APIs prepared for CRM and BI tool integration

### Business Outcomes
- **Lead Generation**: Qualified prospect lists with business context
- **Market Intelligence**: Comprehensive competitive analysis capabilities
- **Sales Efficiency**: 50%+ reduction in prospecting time
- **Data-Driven Decisions**: Analytics-backed territory and service planning

### Data Quality Improvements
- **Validated Detections**: Human-reviewed accuracy of 95%+
- **Rich Context**: Business relationships and waste generation estimates
- **Historical Tracking**: Time-series analysis of market changes
- **Geographic Precision**: Sub-meter accuracy for service planning

---

## Next Immediate Steps

1. **Week 1 Priority**: Set up PostgreSQL with PostGIS and create core schema
2. **Data Migration**: Convert existing JSONL data to database format
3. **API Development**: Build FastAPI foundation with basic CRUD operations
4. **Integration**: Modify existing scanning scripts to use database

This plan transforms the current proof-of-concept into a production-ready business intelligence platform for the waste management industry, building directly on the solid foundation you've already established.

