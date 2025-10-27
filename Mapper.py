class Mapper(object):
  def map_ids(self, targets):
    id_mapping = {'ADJ': 1, 'ADP': 2, 'ADV': 3, 'AUX': 4, 'CCONJ': 5, 'DET': 6,
              'INTJ': 7, 'NOUN': 8, 'NUM': 9, 'PART': 10, 'PRON': 11, 'PROPN': 12,
              'PUNCT': 13, 'SCONJ': 14, 'SYM': 15, 'VERB': 16, 'X': 17, 'UNK': 18}
    targets_map = []
    for target in targets:
        id = id_mapping[target]
        targets_map.append(id)
    return targets_map


  def mapping(self, targets):
    mapped_targets = []
    for target in targets:
        target_map = self.map_ids(target)
        mapped_targets.append(target_map)
    return mapped_targets


  def map_pos(self, targets):
    pos_mapping = {1: 'ADJ', 2: 'ADP', 3: 'ADV', 4: 'AUX', 5: 'CCONJ', 6: 'DET',
                  7: 'INTJ', 8: 'NOUN', 9: 'NUM', 10: 'PART', 11: 'PRON', 12: 'PROPN',
                  13: 'PUNCT', 14: 'SCONJ', 15: 'SYM', 16: 'VERB', 17: 'X', 18: 'UNK'}
    targets_unmap = []
    for target in targets:
        pos = pos_mapping[target]
        targets_unmap.append(pos)
    return targets_unmap


  def unmapping(self, targets):
    unmap_targets = []
    for target in targets:
        target_unmap = self.map_pos(target)
        unmap_targets.append((target_unmap))
    return unmap_targets