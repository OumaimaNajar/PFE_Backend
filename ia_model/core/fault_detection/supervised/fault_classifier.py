class FaultTypeClassifier:
    def __init__(self):
        self.fault_types = {
            'MECHANICAL': {
                'keywords': ['bearing', 'shaft', 'gear', 'belt', 'motor', 'pump', 'vibration', 'noise', 'alignment'],
                'locations': ['BR200', 'BR400', 'BR430', 'MECHASSY', 'MOLD'],
                'confidence_boost': 0.3
            },
            'ELECTRICAL': {
                'keywords': ['power', 'voltage', 'current', 'circuit', 'electrical', 'wire', 'connection', 'battery'],
                'locations': ['COMP310', 'HIGHVOL', 'LOWVOL'],
                'confidence_boost': 0.25
            },
            'HYDRAULIC': {
                'keywords': ['leak', 'pressure', 'fluid', 'hydraulic', 'oil', 'pump', 'valve', 'cylinder'],
                'locations': ['BPM3100', 'L3100'],
                'confidence_boost': 0.2
            },
            'PNEUMATIC': {
                'keywords': ['air', 'pressure', 'pneumatic', 'compressor', 'valve', 'cylinder'],
                'locations': ['SPRAY', 'CLEAN'],
                'confidence_boost': 0.2
            },
            'ELECTRONIC': {
                'keywords': ['sensor', 'control', 'electronic', 'display', 'signal', 'pcb', 'board'],
                'locations': ['MARKING', 'FINALPKG'],
                'confidence_boost': 0.25
            },
            'SOFTWARE': {
                'keywords': ['software', 'program', 'system', 'error', 'code', 'configuration', 'setting'],
                'locations': ['CONTROL', 'SYSTEM'],
                'confidence_boost': 0.15
            }
        }
        
    def predict_fault_type(self, input_data):
        description = str(input_data.get('description', '')).lower()
        location = str(input_data.get('LOCATION', '')).upper()
        
        scores = {}
        for fault_type, config in self.fault_types.items():
            score = 0.0
            
            # Keyword matching
            matched_keywords = [k for k in config['keywords'] if k in description]
            if matched_keywords:
                score += len(matched_keywords) * 0.1
            
            # Location matching
            if location in config['locations']:
                score += config['confidence_boost']
            
            scores[fault_type] = score
        
        # Get the most likely fault type
        predicted_type = max(scores.items(), key=lambda x: x[1])[0]
        confidence = min(0.95, scores[predicted_type] + 0.4)
        
        return {
            "type": predicted_type,
            "confidence": f"{confidence * 100:.2f}%",
            "suggested_actions": self.get_suggested_actions(predicted_type),
            "maximo_codes": self.get_maximo_codes(predicted_type),
            "matched_patterns": {
                "keywords": [k for k in self.fault_types[predicted_type]['keywords'] 
                           if k in description],
                "location_match": location in self.fault_types[predicted_type]['locations']
            }
        }
        
    def get_suggested_actions(self, fault_type):
        actions_map = {
            'MECHANICAL': ['Inspect mechanical components', 'Check alignments'],
            'ELECTRICAL': ['Check power supply', 'Test circuits'],
            'HYDRAULIC': ['Check fluid levels', 'Inspect for leaks'],
            'PNEUMATIC': ['Check air pressure', 'Test pneumatic systems'],
            'ELECTRONIC': ['Test electronic components', 'Check sensors'],
            'SOFTWARE': ['Check system logs', 'Verify software version']
        }
        return actions_map.get(fault_type, [])

    def get_maximo_codes(self, fault_type):
        codes_map = {
            'MECHANICAL': {'problem': 'PB_MEC', 'failure': 'MECH_FAIL'},
            'ELECTRICAL': {'problem': 'PB_ELE', 'failure': 'ELEC_FAIL'},
            'HYDRAULIC': {'problem': 'PB_HYD', 'failure': 'HYDR_FAIL'},
            'PNEUMATIC': {'problem': 'PB_PNE', 'failure': 'PNEU_FAIL'},
            'ELECTRONIC': {'problem': 'PB_ELC', 'failure': 'ELEC_FAIL'},
            'SOFTWARE': {'problem': 'PB_SFT', 'failure': 'SOFT_FAIL'}
        }
        return codes_map.get(fault_type, {})