from DRecPy.Evaluation import LossTracker
import pytest


@pytest.fixture
def loss_tracker():
    return LossTracker()


def test_initial_state(loss_tracker):
    assert loss_tracker.losses == []
    assert loss_tracker.curr_avg_loss == 0
    assert loss_tracker.epoch_losses == []
    assert loss_tracker.curr_avg_epoch_loss == 0
    assert loss_tracker.epoch_callback_results == {}
    assert loss_tracker.called_epochs == []


def test_add_batch_loss_0(loss_tracker):
    loss_tracker.add_batch_loss(2)
    assert loss_tracker.losses == [2]
    assert loss_tracker.curr_avg_loss == 2


def test_add_batch_loss_1(loss_tracker):
    loss_tracker.add_batch_loss(2)
    loss_tracker.add_batch_loss(4)
    assert loss_tracker.losses == [2, 4]
    assert loss_tracker.curr_avg_loss == 3


def test_add_batch_loss_3(loss_tracker):
    loss_tracker.add_batch_loss(2)
    loss_tracker.add_batch_loss(4)
    loss_tracker.add_batch_loss(0)
    assert loss_tracker.losses == [2, 4, 0]
    assert loss_tracker.curr_avg_loss == 2


def test_get_batch_avg_loss(loss_tracker):
    loss_tracker.curr_avg_loss = 3
    assert loss_tracker.get_batch_avg_loss() == 3


def test_reset_batch_losses(loss_tracker):
    loss_tracker.losses = [1, 2]
    loss_tracker.curr_avg_loss = 1.5
    loss_tracker.reset_batch_losses()
    assert loss_tracker.losses == []
    assert loss_tracker.curr_avg_loss == 0


def test_update_epoch_loss_0(loss_tracker):
    loss_tracker.curr_avg_loss = 2
    loss_tracker.update_epoch_loss()
    assert loss_tracker.curr_avg_epoch_loss == 2
    assert loss_tracker.epoch_losses == [2]


def test_update_epoch_loss_1(loss_tracker):
    loss_tracker.curr_avg_loss = 2
    loss_tracker.update_epoch_loss()
    loss_tracker.curr_avg_loss = 4
    loss_tracker.update_epoch_loss()
    assert loss_tracker.curr_avg_epoch_loss == 3
    assert loss_tracker.epoch_losses == [2, 4]


def test_reset_epoch_losses(loss_tracker):
    loss_tracker.curr_avg_epoch_loss = 2
    loss_tracker.epoch_losses = [2]
    loss_tracker.reset_epoch_losses()
    assert loss_tracker.curr_avg_epoch_loss == 0
    assert loss_tracker.epoch_losses == []


def test_add_epoch_callback_result_0(loss_tracker):
    loss_tracker.add_epoch_callback_result('test', 2, 1)
    assert loss_tracker.epoch_callback_results == {'test': [2]}
    assert loss_tracker.called_epochs == [1]


def test_add_epoch_callback_result_1(loss_tracker):
    loss_tracker.add_epoch_callback_result('test', 2, 1)
    loss_tracker.add_epoch_callback_result('test', 3, 2)
    loss_tracker.add_epoch_callback_result('test', 1, 3)
    loss_tracker.add_epoch_callback_result('bla', 5, 1)
    loss_tracker.add_epoch_callback_result('bla', 5, 2)
    loss_tracker.add_epoch_callback_result('bla', 5, 3)
    assert loss_tracker.epoch_callback_results == {'test': [2, 3, 1], 'bla': [5, 5, 5]}
    assert loss_tracker.called_epochs == [1, 2, 3]


def test_display_graph_0(loss_tracker):
    """Test epoch callback results for metric error."""
    loss_tracker.add_epoch_callback_result('test', 2, 1)
    loss_tracker.add_epoch_callback_result('test', 3, 2)
    loss_tracker.add_epoch_callback_result('test', 1, 3)
    loss_tracker.add_epoch_callback_result('bla', 5, 1)
    try:
        loss_tracker.display_graph()
        assert False
    except Exception as e:
        assert str(e) == 'Epoch callback results for metric bla are not defined for all called epochs: number of ' \
                         'called epochs: 3, number of epoch callback results for metric bla: 1'


def test_display_graph_1(loss_tracker):
    """Test epoch callback results for metric no error."""
    loss_tracker.add_epoch_callback_result('test', 2, 1)
    loss_tracker.add_epoch_callback_result('test', 3, 2)
    loss_tracker.add_epoch_callback_result('test', 1, 3)
    loss_tracker.add_epoch_callback_result('bla', 5, 1)
    loss_tracker.add_epoch_callback_result('bla', 5, 2)
    loss_tracker.add_epoch_callback_result('bla', 5, 3)
    loss_tracker.display_graph()
    import matplotlib.pyplot as plt
    plt.close()
    assert True
