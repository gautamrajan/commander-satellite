# Dumpster Detection Dataset Building Plan

## Current Situation Analysis

### What We Have Right Now
- **1,218 total scanned tiles** from satellite imagery
- **76 positive detections** from LLM (6.2% hit rate)
- **61 human reviews completed** with **3 actual dumpsters confirmed**
- **Real performance metrics**:
  - **True Positives**: 3 confirmed dumpsters
  - **False Positives**: 58 incorrect detections  
  - **Precision**: 3/61 = **4.9%** (much better than 0%!)
  - **Coverage**: 61/76 = 80% of positive detections reviewed

### The Real Situation
The LLM detection actually **found real dumpsters** but with high false positive rate:
1. **✅ Detection works**: Found 3 confirmed dumpsters in the scan area
2. **❌ High noise**: 58 false positives for every 3 true positives (~19:1 ratio)
3. **✅ Proof of concept**: The approach can find real dumpsters
4. **🎯 Next step**: Need to reduce false positives and improve precision

---

## Revised Strategy: Build on Success

### Immediate Priorities (This Week)
1. **✅ WORKING**: Use the 3 confirmed dumpsters as positive training examples
2. **✅ BONUS**: Use the 58 false positives as high-quality negative examples  
3. **🎯 EXPAND**: Scan 2-3 more commercial areas to find 10-20 total confirmed dumpsters
4. **🎯 ITERATE**: Quick training cycle with current data to improve precision

### Why This Changes Everything
- **We have ground truth**: 3 confirmed positives + 58 confirmed negatives = 61 labeled examples
- **Pattern exists**: Dumpsters ARE detectable in satellite imagery at zoom 22
- **False positive patterns**: Can train the model what NOT to detect
- **Faster path to dataset**: Don't need to start from scratch

---

## Goal: Build High-Quality Annotated Dataset

### Target Dataset Size
- **Phase 1**: 1,000 confirmed dumpster images with bounding boxes
- **Phase 2**: 5,000 total images (2,500 dumpsters + 2,500 negatives)
- **Phase 3**: 10,000+ images for production model training

### Dataset Requirements
- **Bounding box annotations** (x, y, width, height) for each dumpster
- **Diverse examples**: Different sizes, angles, lighting, seasons
- **Geographic diversity**: Multiple cities/regions
- **Quality validation**: Double-annotation for accuracy

---

## Phase 1: Optimize Current Pipeline & Expand Dataset (Week 1)

### Step 1.1: Leverage What's Working

**Current Success**: 4.9% precision means the LLM CAN find dumpsters, we just need more data and better filtering.

**Immediate Strategy**:
1. **Expand scan area** - we found 3 dumpsters in current area, let's scan more areas
2. **Lower confidence threshold** - might catch more true positives we're missing
3. **Focus on areas near confirmed dumpsters** - similar contexts likely to have more

```python
# Expand successful scanning
python scan_dumpsters.py \
  --tiles_dir new_commercial_area \
  --min_confidence 0.3 \  # Lower threshold to catch more
  --context_radius 1 \    # More context helps accuracy
  --coarse_factor 2 \     # Pre-filter for efficiency
  --log_all all_results_area2.jsonl
```

### Step 1.2: Smart Negative Mining

**Key Insight**: We have 58 high-quality negative examples already reviewed!

**Use the false positives**:
1. **Analyze failure patterns** - what does the LLM incorrectly identify as dumpsters?
2. **Create negative training set** - these 58 images are valuable "hard negatives"
3. **Pattern recognition** - identify common false positive types (cars, shadows, buildings)

### Step 1.2: Build Annotation Pipeline Tools

**Create `annotation_tools.py`**:
```python
class BoundingBoxAnnotator:
    """Tool for efficient bounding box annotation"""
    def __init__(self, image_dir: str, output_file: str):
        self.images = self.load_candidate_images()
        self.annotations = []
    
    def create_annotation_interface(self):
        """Web interface for drawing bounding boxes"""
        # HTML5 canvas for drawing boxes
        # Keyboard shortcuts for efficiency
        # Auto-save annotations
    
    def export_yolo_format(self):
        """Export in YOLO training format"""
        # class_id center_x center_y width height (normalized)
    
    def export_coco_format(self):
        """Export in COCO format for other frameworks"""
```

**Create `dataset_manager.py`**:
```python
class DatasetManager:
    """Manage dataset collection and quality"""
    
    def collect_candidates(self):
        """Systematically collect candidate images"""
        # Sample from different geographic areas
        # Ensure diverse lighting/seasons
        # Include negative examples
    
    def validate_annotations(self):
        """Quality control for annotations"""
        # Check for missing/duplicate boxes
        # Validate box coordinates
        # Flag low-confidence annotations
    
    def create_training_splits(self):
        """Split data for training/validation/test"""
        # Geographic splits to test generalization
        # Balanced class distribution
```

### Step 1.3: Enhanced Review Interface

**Upgrade the Flask app** to support bounding box annotation:

```html
<!-- Enhanced review interface -->
<div class="annotation-interface">
    <canvas id="image-canvas"></canvas>
    <div class="annotation-tools">
        <button id="draw-box">Draw Dumpster Box</button>
        <button id="mark-negative">No Dumpster</button>
        <button id="skip">Skip/Unclear</button>
    </div>
    <div class="annotation-list">
        <!-- List of drawn boxes with confidence scores -->
    </div>
</div>
```

---

## Phase 2: Systematic Data Collection (Week 2)

### Step 2.1: Geographic Targeting Strategy

**Target High-Probability Areas**:
1. **Commercial zones**: Strip malls, office parks, restaurants
2. **Industrial areas**: Warehouses, manufacturing facilities  
3. **Retail locations**: Shopping centers, big box stores
4. **Service businesses**: Auto shops, construction companies

**Implementation**:
```python
# Enhanced imagery collection focusing on commercial areas
python grab_imagery.py \
  --lat 37.7749 --lon -122.4194 \  # San Francisco commercial district
  --area_sqmi 0.5 \
  --zoom 20 \  # Higher resolution for annotation
  --save_tiles_dir commercial_tiles
```

### Step 2.2: Multi-Source Data Collection

**Beyond Current LLM Detection**:
1. **Manual area selection**: Use satellite imagery to manually identify promising areas
2. **Crowdsourced collection**: Tools for contributors to mark dumpster locations
3. **Existing datasets**: Check for publicly available dumpster/waste container datasets
4. **Street view correlation**: Use Google Street View API to validate satellite detections

### Step 2.3: Negative Example Collection

**Critical for training**: Need high-quality negative examples
- Similar objects: Cars, trucks, trailers, storage containers
- Similar contexts: Loading docks, parking areas, industrial zones
- Edge cases: Partial occlusions, shadows, different lighting

---

## Phase 3: Annotation Workflow Optimization (Week 3)

### Step 3.1: Efficient Annotation Interface

**Key Features**:
- **Keyboard shortcuts**: Space (next), Enter (confirm), Esc (cancel box)
- **Auto-suggestions**: Pre-populate likely boxes using current detections
- **Batch operations**: Annotate similar images together
- **Progress tracking**: Clear progress indicators and session management

**Annotation Speed Targets**:
- Simple images: 30 seconds each
- Complex images: 60-90 seconds each
- Target: 50-100 annotations per hour

### Step 3.2: Quality Assurance Process

**Double Annotation**:
- 20% of images annotated by multiple people
- Flag disagreements for expert review
- Measure inter-annotator agreement

**Validation Checks**:
- Minimum/maximum box sizes
- Box overlap detection
- Obvious false positives (sky, water, etc.)

### Step 3.3: Annotation Guidelines

**Create clear guidelines for annotators**:
```markdown
# Dumpster Annotation Guidelines

## What to Annotate
- Commercial dumpsters (front-load, rear-load)
- Roll-off containers
- Large waste containers (>2 cubic yards)

## What NOT to Annotate  
- Residential trash cans
- Recycling bins
- Personal vehicles
- Trailers/RVs (unless clearly being used for waste)

## Bounding Box Rules
- Include the entire visible dumpster
- Include wheels/legs if visible
- Don't include shadows unless part of the dumpster
- For partially occluded dumpsters, include the visible portion
```

---

## Phase 4: Dataset Enhancement & Validation (Week 4)

### Step 4.1: Data Augmentation Strategy

**Automated augmentation** to increase dataset size:
- Rotation (±15 degrees)
- Brightness/contrast variations
- Weather effects (simulate seasons)
- Resolution scaling (train on multiple scales)

### Step 4.2: Dataset Statistics & Analysis

**Build dataset analysis tools**:
```python
def analyze_dataset(annotations_file):
    """Generate dataset statistics"""
    stats = {
        'total_images': len(images),
        'total_annotations': len(annotations),
        'avg_dumpsters_per_image': avg_annotations,
        'size_distribution': box_size_histogram,
        'geographic_distribution': location_map,
        'false_positive_rate': validation_accuracy
    }
    return stats
```

### Step 4.3: Initial Model Training & Validation

**Quick validation with small model**:
- Train YOLOv8 nano on first 500 annotations
- Evaluate on held-out test set
- Identify common failure modes
- Use results to improve annotation guidelines

---

## Implementation Tools & Scripts

### Tool 1: Enhanced Scanning for Dataset Collection
```bash
# scan_for_dataset.py - Modified scanner optimized for dataset building
python scan_for_dataset.py \
  --commercial_areas commercial_zones.geojson \
  --output_dir dataset_candidates \
  --sample_rate 0.1 \  # Sample 10% of tiles to avoid overwhelming annotators
  --min_confidence 0.2  # Cast wide net
```

### Tool 2: Annotation Interface
```bash
# annotation_server.py - Web interface for efficient annotation
python annotation_server.py \
  --image_dir dataset_candidates \
  --port 5001
# Opens interface at http://localhost:5001
```

### Tool 3: Dataset Management
```bash
# dataset_tools.py - Command line tools for dataset management
python dataset_tools.py export --format yolo --output training_data/
python dataset_tools.py validate --check-duplicates --check-quality
python dataset_tools.py split --train 0.7 --val 0.2 --test 0.1
```

### Tool 4: Quick Model Validation
```bash
# train_validation_model.py - Quick model to test dataset quality
python train_validation_model.py \
  --dataset training_data/ \
  --model yolov8n \
  --epochs 50 \
  --validate
```

---

## Success Metrics & Milestones

### Week 1 Milestones
- [ ] Enhanced scanning pipeline collecting better candidates
- [ ] Basic annotation interface operational
- [ ] 100 images annotated with bounding boxes

### Week 2 Milestones  
- [ ] Geographic targeting producing higher hit rates
- [ ] 500 images annotated
- [ ] Quality assurance process established

### Week 3 Milestones
- [ ] 1,000 images annotated (target for Phase 1)
- [ ] Annotation guidelines refined based on experience
- [ ] Inter-annotator agreement >90%

### Week 4 Milestones
- [ ] Dataset analysis complete
- [ ] Initial model trained and validated
- [ ] Plan for Phase 2 (5,000 images) ready

### Quality Targets
- **Annotation accuracy**: >95% precision on validation set
- **Dataset diversity**: Images from 3+ different cities/regions
- **Annotation speed**: <60 seconds average per image
- **Model validation**: >80% mAP@0.5 on initial test model

---

## Next Immediate Actions

### Today (High Priority)
1. **Extract confirmed examples**: Export the 3 confirmed dumpster tiles and 58 negative examples
2. **Scan more areas**: Target 2-3 new commercial areas near the successful detections
3. **Quick bounding box annotation**: Draw boxes on the 3 confirmed dumpsters

### This Week
1. **Build simple training pipeline**: Train YOLOv8 nano on current 61 examples
2. **Test improved model**: Run it on unscanned tiles to see if precision improves
3. **Expand dataset**: Review new scan results to find 5-10 more confirmed dumpsters

### Success Metrics (Week 1)
- [ ] 10+ confirmed dumpster detections with bounding boxes
- [ ] 100+ confirmed negative examples  
- [ ] Initial YOLOv8 model trained and tested
- [ ] Precision improvement measured (target: >10% vs current 4.9%)

This approach leverages the success you've already achieved and builds on real, confirmed examples rather than starting from scratch.
