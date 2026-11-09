"""CPU semantics only: a greedy verified prefix, not a forced acceptance."""
def greedy_reference(draft,target,bonus):
    result=[]
    for d,t in zip(draft,target):
        result.append(t)
        if d!=t:return result
    return result+[bonus]

def capped_reference(draft,target,bonus,new_gamma):
    return greedy_reference(draft[:new_gamma],target[:new_gamma],target[new_gamma])
