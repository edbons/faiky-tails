import torch
import numpy
from transformers import GPT2Tokenizer


def get_example_inputs(prompt_text: str="", tokenizer: GPT2Tokenizer=None, num_attention_heads: int=1, hidden_size: int=1, num_layer: int=1, device='cpu'):
    context = tokenizer.encode(prompt_text)

    septok = tokenizer.convert_tokens_to_ids('[SEP]')
    starttok = tokenizer.convert_tokens_to_ids('<s>')
    endkeytok = tokenizer.convert_tokens_to_ids('_endkw_')
    context = [starttok] + context + [endkeytok] + [septok]
    
    input_ids = torch.LongTensor(context)
    input_ids = torch.unsqueeze(input_ids, 0)

    #Empty Past State for generating first word
    empty_past = []
    batch_size = input_ids.size(0)

    past_shape = [2, batch_size, num_attention_heads, 0, hidden_size // num_attention_heads]
    for _ in range(num_layer):
        empty_past.append(torch.empty(past_shape).type(torch.float32).to(device))
       
    return input_ids, empty_past


def update(output, batch_size, beam_size, device, **kwargs):
    """
    Update the inputs for next inference.
    """
    num_layer = kwargs['num_layer']
    last_state = (torch.from_numpy(output[0]).to(device)
                        if isinstance(output[0], numpy.ndarray) else output[0].clone().detach().cpu())

    input_ids = last_state.view(batch_size * beam_size, -1).to(device)

    input_unfinished_sents_id = -2

    beam_select_idx = (torch.from_numpy(output[input_unfinished_sents_id - 2]).to(device) if isinstance(
        output[input_unfinished_sents_id - 2], numpy.ndarray) else output[input_unfinished_sents_id - 2].clone().detach().to(device))
    
    input_log_probs = (torch.from_numpy(output[input_unfinished_sents_id - 1]).to(device) if isinstance(
        output[input_unfinished_sents_id - 1], numpy.ndarray) else output[input_unfinished_sents_id - 1].clone().detach().to(device))
    
    input_unfinished_sents = (torch.from_numpy(output[input_unfinished_sents_id]).to(device) if isinstance(
        output[input_unfinished_sents_id], numpy.ndarray) else
                                    output[input_unfinished_sents_id].clone().detach().to(device))
    prev_step_scores = (torch.from_numpy(output[-1]).to(device)
                                if isinstance(output[-1], numpy.ndarray) else output[-1].clone().detach().to(device))

    past = []
    if isinstance(output[1], tuple):  # past in torch output is tuple
        past = list(output[1])
    else:
        for i in range(num_layer):
            past_i = (torch.from_numpy(output[i + 1])
                        if isinstance(output[i + 1], numpy.ndarray) else output[i + 1].clone().detach())
            past.append(past_i.to(device)) 

    inputs = {
        'input_ids': input_ids,
        'beam_select_idx': beam_select_idx,
        'input_log_probs': input_log_probs,
        'input_unfinished_sents': input_unfinished_sents,
        'prev_step_scores': prev_step_scores,
    }
    ort_inputs = {
        'input_ids': numpy.ascontiguousarray(input_ids.cpu().numpy()),
        'beam_select_idx': numpy.ascontiguousarray(beam_select_idx.cpu().numpy()),
        'input_log_probs': numpy.ascontiguousarray(input_log_probs.cpu().numpy()),
        'input_unfinished_sents': numpy.ascontiguousarray(input_unfinished_sents.cpu().numpy()),
        'prev_step_scores': numpy.ascontiguousarray(prev_step_scores.cpu().numpy()),
    }
    for i, past_i in enumerate(past):
        ort_inputs[f'past_{i}'] = numpy.ascontiguousarray(past_i.cpu().numpy())
    
    return inputs, ort_inputs, past


def generate(tokenizer, input_text: list, ort_session = None, num_tokens_to_produce: int=30, device: str="cpu", **kwargs) -> list:
    input_text = " _kw_ ".join(input_text)

    input_ids, past = get_example_inputs(input_text, tokenizer=tokenizer, device=device, **kwargs)

    beam_select_idx = torch.zeros([1, input_ids.shape[0]]).long()
    input_log_probs = torch.zeros([input_ids.shape[0], 1])
    input_unfinished_sents = torch.ones([input_ids.shape[0], 1], dtype=torch.bool)
    prev_step_scores = torch.zeros([input_ids.shape[0], 1])

    ort_inputs = {
        'input_ids': numpy.ascontiguousarray(input_ids.cpu().numpy()),       
        'beam_select_idx': numpy.ascontiguousarray(beam_select_idx.cpu().numpy()),
        'input_log_probs': numpy.ascontiguousarray(input_log_probs.cpu().numpy()),
        'input_unfinished_sents': numpy.ascontiguousarray(input_unfinished_sents.cpu().numpy()),       
        'prev_step_scores': numpy.ascontiguousarray(prev_step_scores.cpu().numpy())
    }
    
    for i, past_i in enumerate(past):
        ort_inputs[f'past_{i}'] = numpy.ascontiguousarray(past_i.cpu().numpy())
    
    batch_size = input_ids.size(0)
    beam_size = 4

    for _ in range(num_tokens_to_produce):
        outputs = ort_session.run(None, ort_inputs)
        inputs, ort_inputs, past = update(outputs, batch_size, beam_size, device, **kwargs)

        if not inputs['input_unfinished_sents'].any():
            break
    
    result = [tokenizer.decode(inputs['input_ids'][i], skip_special_tokens=False, clean_up_tokenization_spaces=False) for i in range(inputs['input_ids'].shape[0])]
    return result

if __name__ == '__main__':
    pass