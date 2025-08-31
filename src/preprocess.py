
import os
import json
import numpy as np
import random
import torch
from torch_geometric.data import Data

def parse_patient_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Extract basic features from FHIR Patient resource
    birthdate = data.get('birthDate', '')
    gender = data.get('gender', '')
    race = ''
    ethnicity = ''
    # Synthea FHIR sometimes encodes race/ethnicity in extensions
    for ext in data.get('extension', []):
        url = ext.get('url', '')
        if 'us-core-race' in url:
            race = ext.get('valueString', '')
        if 'us-core-ethnicity' in url:
            ethnicity = ext.get('valueString', '')
    return {
        'birthdate': birthdate,
        'gender': gender,
        'race': race,
        'ethnicity': ethnicity
    }

def build_graph(data_dir):
    """
    Build a graph from FHIR data in the specified directory
    Returns a PyTorch Geometric Data object
    """
    # Find all patient JSON files
    files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    patients = {}
    encounters = {}
    observations = {}
    conditions = {}  # Added conditions
    procedures = {}  # Added procedures
    
    patient_features = []
    encounter_features = []
    observation_features = []
    condition_features = []  # Added condition features
    procedure_features = []  # Added procedure features
    
    patient_nodes = []
    encounter_nodes = []
    observation_nodes = []
    condition_nodes = []  # Added condition nodes
    procedure_nodes = []  # Added procedure nodes
    
    edge_index = [[], []]  # [source, target]
    node_type = []  # 0: patient, 1: encounter, 2: observation, 3: condition, 4: procedure
    
    # Parse all files
    for f in files:
        if 'hospitalInformation' in f or 'practitionerInformation' in f:
            continue
        path = os.path.join(data_dir, f)
        with open(path, 'r', encoding='utf-8') as fp:
            try:
                data = json.load(fp)
            except json.JSONDecodeError:
                print(f"Error parsing {path}, skipping...")
                continue
                
        resourceType = data.get('resourceType', '')
        
        # Process by resource type
        if resourceType == 'Patient':
            pid = data.get('id', f)
            patients[pid] = data
            features = parse_patient_json(path)
            patient_features.append(features)
            patient_nodes.append(pid)
            node_type.append(0)  # 0 for patient
            
        elif resourceType == 'Encounter':
            eid = data.get('id', f)
            encounters[eid] = data
            encounter_features.append({
                'class': data.get('class', {}).get('code', ''),
                'type': data.get('type', [{}])[0].get('coding', [{}])[0].get('code', ''),
                'status': data.get('status', '')
            })
            encounter_nodes.append(eid)
            node_type.append(1)  # 1 for encounter
            
        elif resourceType == 'Observation':
            oid = data.get('id', f)
            observations[oid] = data
            
            # Extract code and value - handle different value types
            value = ''
            if 'valueQuantity' in data:
                value = str(data.get('valueQuantity', {}).get('value', ''))
                unit = data.get('valueQuantity', {}).get('unit', '')
                if unit:
                    value += f" {unit}"
            elif 'valueCodeableConcept' in data:
                value = data.get('valueCodeableConcept', {}).get('coding', [{}])[0].get('code', '')
            elif 'valueString' in data:
                value = data.get('valueString', '')
            
            observation_features.append({
                'code': data.get('code', {}).get('coding', [{}])[0].get('code', ''),
                'display': data.get('code', {}).get('coding', [{}])[0].get('display', ''),
                'value': value,
                'status': data.get('status', '')
            })
            observation_nodes.append(oid)
            node_type.append(2)  # 2 for observation
            
        elif resourceType == 'Condition':
            cid = data.get('id', f)
            conditions[cid] = data
            condition_features.append({
                'code': data.get('code', {}).get('coding', [{}])[0].get('code', ''),
                'display': data.get('code', {}).get('coding', [{}])[0].get('display', ''),
                'verification': data.get('verificationStatus', {}).get('coding', [{}])[0].get('code', ''),
                'category': data.get('category', [{}])[0].get('coding', [{}])[0].get('code', '')
            })
            condition_nodes.append(cid)
            node_type.append(3)  # 3 for condition
            
        elif resourceType == 'Procedure':
            pid = data.get('id', f)
            procedures[pid] = data
            # Check if bodysite is present or not
            bodysite = data.get('bodySite', [])
            has_bodysite = 1 if bodysite else 0
            
            procedure_features.append({
                'code': data.get('code', {}).get('coding', [{}])[0].get('code', ''),
                'display': data.get('code', {}).get('coding', [{}])[0].get('display', ''),
                'status': data.get('status', ''),
                'has_bodysite': has_bodysite  # Flag for bodysite presence
            })
            procedure_nodes.append(pid)
            node_type.append(4)  # 4 for procedure
    
    # Build node features
    
    # Patients
    genders = sorted(list(set(p['gender'] for p in patient_features)))
    races = sorted(list(set(p['race'] for p in patient_features)))
    ethnicities = sorted(list(set(p['ethnicity'] for p in patient_features)))
    px = []
    for p in patient_features:
        birth_year = int(p['birthdate'][:4]) if p['birthdate'] else 0
        gender_vec = [int(p['gender'] == g) for g in genders]
        race_vec = [int(p['race'] == r) for r in races]
        eth_vec = [int(p['ethnicity'] == e) for e in ethnicities]
        px.append([birth_year] + gender_vec + race_vec + eth_vec)
    
    # Encounters
    classes = sorted(list(set(e['class'] for e in encounter_features)))
    types = sorted(list(set(e['type'] for e in encounter_features)))
    statuses = sorted(list(set(e['status'] for e in encounter_features)))
    ex = []
    for e in encounter_features:
        class_vec = [int(e['class'] == c) for c in classes]
        type_vec = [int(e['type'] == t) for t in types]
        status_vec = [int(e['status'] == s) for s in statuses]
        ex.append(class_vec + type_vec + status_vec)
    
    # Observations
    codes = sorted(list(set(o['code'] for o in observation_features)))
    statuses = sorted(list(set(o['status'] for o in observation_features)))
    ox = []
    for o in observation_features:
        code_vec = [int(o['code'] == c) for c in codes]
        status_vec = [int(o['status'] == s) for s in statuses]
        # Try to convert value to float for quantitative observations
        try:
            value = float(o['value'].split()[0]) if o['value'] else 0.0
        except (ValueError, TypeError):
            value = 0.0
        ox.append(code_vec + status_vec + [value])
    
    # Conditions
    condition_codes = sorted(list(set(c['code'] for c in condition_features)))
    verification_statuses = sorted(list(set(c['verification'] for c in condition_features)))
    categories = sorted(list(set(c['category'] for c in condition_features)))
    cx = []
    for c in condition_features:
        code_vec = [int(c['code'] == code) for code in condition_codes]
        verification_vec = [int(c['verification'] == vs) for vs in verification_statuses]
        category_vec = [int(c['category'] == cat) for cat in categories]
        cx.append(code_vec + verification_vec + category_vec)
    
    # Procedures
    procedure_codes = sorted(list(set(p['code'] for p in procedure_features)))
    procedure_statuses = sorted(list(set(p['status'] for p in procedure_features)))
    px_proc = []
    for p in procedure_features:
        code_vec = [int(p['code'] == code) for code in procedure_codes]
        status_vec = [int(p['status'] == s) for s in procedure_statuses]
        # Add bodysite flag
        bodysite_flag = [p['has_bodysite']]
        px_proc.append(code_vec + status_vec + bodysite_flag)
    
    # Combine all node features
    all_features = px + ex + ox + cx + px_proc
    
    if not all_features:
        # No valid nodes or features found, return empty Data object
        print("No valid features found. Returning empty graph.")
        x = np.zeros((0, 0), dtype=np.float32)
        edge_index = np.zeros((2, 0), dtype=np.int64)
        return Data(x=x, edge_index=edge_index)
    
    # Ensure all feature vectors have the same length
    max_len = max(len(feat) for feat in all_features)
    all_features = [feat + [0] * (max_len - len(feat)) for feat in all_features]
    
    x = np.array(all_features, dtype=np.float32)
    
    # Build edges: 
    # patient->encounter, encounter->observation, encounter->condition, encounter->procedure
    all_nodes = patient_nodes + encounter_nodes + observation_nodes + condition_nodes + procedure_nodes
    node_id_map = {nid: i for i, nid in enumerate(all_nodes)}
    
    # Patient->Encounter
    for eid, enc in encounters.items():
        pid = enc.get('subject', {}).get('reference', '').replace('Patient/', '')
        if pid in node_id_map and eid in node_id_map:
            edge_index[0].append(node_id_map[pid])
            edge_index[1].append(node_id_map[eid])
    
    # Encounter->Observation
    for oid, obs in observations.items():
        eid = obs.get('encounter', {}).get('reference', '').replace('Encounter/', '')
        if eid in node_id_map and oid in node_id_map:
            edge_index[0].append(node_id_map[eid])
            edge_index[1].append(node_id_map[oid])
    
    # Encounter->Condition
    for cid, cond in conditions.items():
        eid = cond.get('encounter', {}).get('reference', '').replace('Encounter/', '')
        if eid in node_id_map and cid in node_id_map:
            edge_index[0].append(node_id_map[eid])
            edge_index[1].append(node_id_map[cid])
    
    # Encounter->Procedure
    for pid, proc in procedures.items():
        eid = proc.get('encounter', {}).get('reference', '').replace('Encounter/', '')
        if eid in node_id_map and pid in node_id_map:
            edge_index[0].append(node_id_map[eid])
            edge_index[1].append(node_id_map[pid])
    
    # Create condition->procedure edges for diagnostic procedures
    # This represents the relationship between conditions and their associated procedures
    for pid, proc in procedures.items():
        # Check if the procedure has a reason reference pointing to a condition
        reason_refs = proc.get('reasonReference', [])
        for reason in reason_refs:
            cid = reason.get('reference', '').replace('Condition/', '')
            if cid in node_id_map and pid in node_id_map:
                edge_index[0].append(node_id_map[cid])
                edge_index[1].append(node_id_map[pid])
    
    # Convert to PyTorch tensors
    if edge_index[0]:  # Check if edges exist
        edge_index = torch.tensor(edge_index, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    x = torch.tensor(x, dtype=torch.float)
    
    # Create PyTorch Geometric Data object
    data = Data(
        x=x, 
        edge_index=edge_index,
        node_type=torch.tensor(node_type, dtype=torch.long),
    )
    
    # Store metadata for later use
    data.num_nodes = x.shape[0]
    data.num_node_features = x.shape[1]
    
    return data

def introduce_anomalies(graph_data, anomaly_ratio=0.05, anomaly_types=None):
    """
    Introduce synthetic anomalies to graph data for evaluation
    
    Args:
        graph_data: PyTorch Geometric Data object
        anomaly_ratio: Percentage of nodes to introduce anomalies to
        anomaly_types: List of anomaly types to introduce
    
    Returns:
        Modified graph data and ground truth anomaly labels
    """
    if not anomaly_types:
        # Default anomaly types:
        # 1. VC: Observation Value/Code Mismatch
        # 2. PD: Implausible Procedure for Diagnosis
        # 3. DD: Bodysite for Surgery (Missing)
        anomaly_types = ['VC', 'PD', 'DD']
    
    # Convert to numpy for easier manipulation
    x = graph_data.x.clone()
    edge_index = graph_data.edge_index.clone()
    node_types = graph_data.node_type.clone() if hasattr(graph_data, 'node_type') else None
    
    # Initialize anomaly labels (0 = normal, 1 = anomaly)
    anomaly_labels = torch.zeros(x.shape[0], dtype=torch.int)
    
    # Calculate number of nodes to modify
    num_anomalies = max(int(x.shape[0] * anomaly_ratio), 1)
    print(f"Introducing {num_anomalies} anomalies ({anomaly_ratio*100:.1f}% of nodes)")
    
    # Randomly select nodes to introduce anomalies
    anomaly_indices = random.sample(range(x.shape[0]), num_anomalies)
    
    for idx in anomaly_indices:
        # Mark as anomaly
        anomaly_labels[idx] = 1
        
        # Determine anomaly type based on node type (if available) or random selection
        if node_types is not None:
            node_type = node_types[idx].item()
        else:
            node_type = random.randint(0, 2)  # Random type if not available
        
        # Introduce anomalies based on type
        anomaly_type = random.choice(anomaly_types)
        
        if anomaly_type == 'VC':  # Observation Value/Code Mismatch
            # Replace values with implausible ones or mix up codes
            if node_type == 2:  # Observation
                # Get a random feature index - typically the last few are values
                feat_idx = random.randint(x.shape[1] - 5, x.shape[1] - 1)
                # Replace with an implausible value (e.g., multiply by 10)
                x[idx, feat_idx] = x[idx, feat_idx] * 10
            
        elif anomaly_type == 'PD':  # Implausible Procedure for Diagnosis
            # For procedures (type 4) or conditions (type 3)
            if node_type in [3, 4]:
                # Scramble feature values
                feat_start = random.randint(0, x.shape[1] - 10)
                feat_end = feat_start + random.randint(3, 8)
                # Swap values
                for f in range(feat_start, min(feat_end, x.shape[1])):
                    x[idx, f] = 1 - x[idx, f]  # Invert binary features
        
        elif anomaly_type == 'DD':  # Missing Bodysite for Surgery
            # For procedures (type 4)
            if node_type == 4:
                # Find bodysite feature (last one in our encoding)
                x[idx, -1] = 0  # Set bodysite flag to 0 (missing)
        
        else:  # General anomaly - introduce noise
            # Add random noise to several features
            num_features = random.randint(1, 5)
            for _ in range(num_features):
                feat_idx = random.randint(0, x.shape[1] - 1)
                if random.random() > 0.5:
                    # Add noise to continuous features
                    x[idx, feat_idx] += random.uniform(-2, 2)
                else:
                    # Flip binary features
                    if x[idx, feat_idx] <= 1:
                        x[idx, feat_idx] = 1 - x[idx, feat_idx]
    
    # Create new graph data with anomalies
    modified_data = Data(
        x=x,
        edge_index=edge_index,
        node_type=node_types
    )
    
    # Copy any additional attributes
    for key, value in graph_data:
        if key not in ['x', 'edge_index', 'node_type']:
            modified_data[key] = value
    
    return modified_data, anomaly_labels

def split_data(graph, test_size=0.2):
    """
    Split graph data into training and testing sets
    
    Args:
        graph: PyTorch Geometric Data object
        test_size: Fraction of nodes to use for testing
    
    Returns:
        train_graph, test_graph
    """
    # Clone the graph
    x = graph.x.clone()
    edge_index = graph.edge_index.clone()
    
    # Determine split point
    num_nodes = x.shape[0]
    num_test = int(num_nodes * test_size)
    
    # Generate random indices
    indices = list(range(num_nodes))
    random.shuffle(indices)
    
    test_indices = indices[:num_test]
    train_indices = indices[num_test:]
    
    # Create masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_indices] = True
    test_mask[test_indices] = True
    
    # Create train/test graphs
    # For simplicity, we keep the same edge structure but mask node features
    train_graph = Data(
        x=x,
        edge_index=edge_index,
        train_mask=train_mask,
        test_mask=test_mask
    )
    
    test_graph = Data(
        x=x,
        edge_index=edge_index,
        train_mask=train_mask,
        test_mask=test_mask
    )
    
    # Copy any additional attributes
    for key, value in graph:
        if key not in ['x', 'edge_index', 'train_mask', 'test_mask']:
            train_graph[key] = value
            test_graph[key] = value
    
    return train_graph, test_graph

if __name__ == "__main__":
    from constants import DATA_DIR
    
    graph = build_graph(DATA_DIR)
    print(f"Original graph: {graph}")
    
    # Test data splitting
    train_data, test_data = split_data(graph, test_size=0.2)
    print(f"Train data: {train_data}")
    print(f"Test data: {test_data}")
    
    # Test anomaly introduction
    test_data_with_anomalies, anomaly_labels = introduce_anomalies(test_data, anomaly_ratio=0.05)
    print(f"Test data with anomalies: {test_data_with_anomalies}")
    print(f"Number of anomalies introduced: {anomaly_labels.sum().item()}")
