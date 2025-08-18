
import os
import json
import numpy as np
from torch_geometric.data import Data

def parse_patient_json(filepath):
    with open(filepath, 'r') as f:
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
    # Find all patient JSON files
    files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    patients = {}
    encounters = {}
    observations = {}
    patient_features = []
    encounter_features = []
    observation_features = []
    patient_nodes = []
    encounter_nodes = []
    observation_nodes = []
    edge_index = [[], []]  # [source, target]
    node_type = []
    # Parse all files
    for f in files:
        if 'hospitalInformation' in f or 'practitionerInformation' in f:
            continue
        path = os.path.join(data_dir, f)
        with open(path, 'r') as fp:
            data = json.load(fp)
        resourceType = data.get('resourceType', '')
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
                'type': data.get('type', [{}])[0].get('coding', [{}])[0].get('code', '')
            })
            encounter_nodes.append(eid)
            node_type.append(1)  # 1 for encounter
        elif resourceType == 'Observation':
            oid = data.get('id', f)
            observations[oid] = data
            observation_features.append({
                'code': data.get('code', {}).get('coding', [{}])[0].get('code', ''),
                'value': str(data.get('valueQuantity', {}).get('value', ''))
            })
            observation_nodes.append(oid)
            node_type.append(2)  # 2 for observation
    # Build node features
    # Patients
    genders = list(set(p['gender'] for p in patient_features))
    races = list(set(p['race'] for p in patient_features))
    ethnicities = list(set(p['ethnicity'] for p in patient_features))
    px = []
    for p in patient_features:
        birth_year = int(p['birthdate'][:4]) if p['birthdate'] else 0
        gender_vec = [int(p['gender'] == g) for g in genders]
        race_vec = [int(p['race'] == r) for r in races]
        eth_vec = [int(p['ethnicity'] == e) for e in ethnicities]
        px.append([birth_year] + gender_vec + race_vec + eth_vec)
    # Encounters
    classes = list(set(e['class'] for e in encounter_features))
    types = list(set(e['type'] for e in encounter_features))
    ex = []
    for e in encounter_features:
        class_vec = [int(e['class'] == c) for c in classes]
        type_vec = [int(e['type'] == t) for t in types]
        ex.append(class_vec + type_vec)
    # Observations
    codes = list(set(o['code'] for o in observation_features))
    ox = []
    for o in observation_features:
        code_vec = [int(o['code'] == c) for c in codes]
        value = float(o['value']) if o['value'] else 0.0
        ox.append(code_vec + [value])
    # Combine all node features
    x = np.array(px + ex + ox, dtype=np.float32)
    # Build edges: patient->encounter, encounter->observation
    node_id_map = {nid: i for i, nid in enumerate(patient_nodes + encounter_nodes + observation_nodes)}
    # Patient->Encounter
    for eid, enc in encounters.items():
        pid = enc.get('subject', {}).get('reference', '').replace('Patient/', '')
        if pid in node_id_map:
            edge_index[0].append(node_id_map[pid])
            edge_index[1].append(node_id_map[eid])
    # Encounter->Observation
    for oid, obs in observations.items():
        eid = obs.get('encounter', {}).get('reference', '').replace('Encounter/', '')
        if eid in node_id_map:
            edge_index[0].append(node_id_map[eid])
            edge_index[1].append(node_id_map[oid])
    edge_index = np.array(edge_index, dtype=np.int64)
    data = Data(x=x, edge_index=edge_index)
    return data

if __name__ == "__main__":
    graph = build_graph(".data")
    print(graph)
