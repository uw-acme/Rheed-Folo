"""The beta SDK's byte-transfer completion must also finalize server content."""
import pytest
from dataerai_console_api import DataeraiConsoleAPI, DataeraiConsoleError


def api_with_content(status, content_id='content'):
    api=object.__new__(DataeraiConsoleAPI)
    calls=[]
    state={'status':status}
    def request(method,path,**kwargs):
        calls.append((method,path))
        if method=='POST':
            state['status']='available'
            return {'status':'completed'}
        return {'current_content':{'id':content_id,'status':state['status']}}
    api._request=request
    return api,calls


def test_incomplete_transfer_is_finalized_through_normal_api():
    api,calls=api_with_content('uploading')
    api.ensure_upload_complete('asset','content','transfer')
    assert calls==[('GET','/api/assets/asset/'),('POST','/api/transfers/transfer/complete/'),('GET','/api/assets/asset/')]


def test_available_content_needs_no_finalization():
    api,calls=api_with_content('available')
    api.ensure_upload_complete('asset','content','transfer')
    assert len(calls)==1


@pytest.mark.parametrize('status,content_id',[('quarantined','content'),('failed','content'),('available','another-version')])
def test_unavailable_or_different_content_cannot_be_reported_saved(status,content_id):
    api,calls=api_with_content(status,content_id)
    with pytest.raises(DataeraiConsoleError):
        api.ensure_upload_complete('asset','content','transfer')
    assert all(method!='POST' for method,_ in calls)


def test_unavailable_upload_is_not_added_to_saved_asset_index(tmp_path):
    from types import SimpleNamespace
    from tests.helpers import make_tracker
    from dataerai_console_api import DataeraiConsoleError
    tracker,client,_=make_tracker(tmp_path)
    previous=len(tracker.asset_index)
    original=client.upload
    def upload(*args,**kwargs):
        result=original(*args,**kwargs)
        return SimpleNamespace(asset_id=result.asset_id,content_id='version',transfer_id='transfer')
    client.upload=upload
    class Console:
        def ensure_upload_complete(self,*args):
            raise DataeraiConsoleError('content is not available')
    tracker._console_api=Console()
    with pytest.raises(DataeraiConsoleError,match='not available'):
        tracker.record_cell(source='x = 1',assigned_names=['x'],user_ns={'x':1})
    assert len(tracker.asset_index)==previous
    assert (tracker.run_dir/'cell-0001/execution-local.json').exists()
