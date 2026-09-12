from copy import deepcopy
import r2_agent

def test_incomplete_numeric_reply_retains_tool_opportunity(monkeypatch):
    replies=iter(['unfinished numeric derivation','```python\nprint(2)\n```','FINAL_ANSWER: 2'])
    class Client:
        def __init__(self):self.calls=[]
        def chat(self,**kwargs):
            # Capture the request at send time, not a later-mutated conversation.
            self.calls.append(deepcopy(kwargs))
            return next(replies)
    monkeypatch.setattr(r2_agent,'run_math',lambda code:{'ok':True,'stdout':'2'})
    c=Client();out=r2_agent.ReasoningAgent(c).solve('calculate',{})
    followup=c.calls[1]['messages'][-1]['content']
    assert 'short Python block' in followup and 'No code.' not in followup
    assert out['final_response']=='FINAL_ANSWER: 2'
    assert len(c.calls)==3
