from collections import OrderedDict

from easy_tpp.runner.base_runner import Runner
from easy_tpp.utils import RunnerPhase, logger, MetricsHelper, MetricsTracker, concat_element, save_pickle
from easy_tpp.utils.const import Backend


@Runner.register(name='std_tpp')
class TPPRunner(Runner):
    """Standard TPP runner
    """

    def __init__(self, runner_config, unique_model_dir=False, **kwargs):
        super(TPPRunner, self).__init__(runner_config, unique_model_dir, **kwargs)

        self.metrics_tracker = MetricsTracker()
        if self.runner_config.trainer_config.metrics is not None:
            self.metric_functions = self.runner_config.get_metric_functions()

        self._init_model()

        pretrain_dir = self.runner_config.model_config.pretrained_model_dir
        if pretrain_dir is not None:
            self._load_model(pretrain_dir)

    def _init_model(self):
        """Initialize the model.
        """
        self.use_torch = self.runner_config.base_config.backend == Backend.Torch

        if self.use_torch:
            from easy_tpp.utils import set_seed
            from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel
            from easy_tpp.torch_wrapper import TorchModelWrapper
            from easy_tpp.utils import count_model_params
            set_seed(self.runner_config.trainer_config.seed)

            self.model = TorchBaseModel.generate_model_from_config(model_config=self.runner_config.model_config)
            self.model_wrapper = TorchModelWrapper(self.model,
                                                   self.runner_config.base_config,
                                                   self.runner_config.model_config,
                                                   self.runner_config.trainer_config)
            num_params = count_model_params(self.model)

        else:
            from easy_tpp.utils.tf_utils import set_seed
            from easy_tpp.model.tf_model.tf_basemodel import TfBaseModel
            from easy_tpp.tf_wrapper import TfModelWrapper
            from easy_tpp.utils.tf_utils import count_model_params
            set_seed(self.runner_config.trainer_config.seed)

            self.model = TfBaseModel.generate_model_from_config(model_config=self.runner_config.model_config)
            self.model_wrapper = TfModelWrapper(self.model,
                                                self.runner_config.base_config,
                                                self.runner_config.model_config,
                                                self.runner_config.trainer_config)
            num_params = count_model_params()

        info_msg = f'Num of model parameters {num_params}'
        logger.info(info_msg)

    def _save_model(self, model_dir, **kwargs):
        """Save the model.

        Args:
            model_dir (str): the dir for model to save.
        """
        if model_dir is None:
            model_dir = self.runner_config.base_config.specs['saved_model_dir']
        self.model_wrapper.save(model_dir)
        logger.critical(f'Save model to {model_dir}')
        return

    def _load_model(self, model_dir, **kwargs):
        """Load the model from the dir.

        Args:
            model_dir (str): the dir for model to load.
        """
        self.model_wrapper.restore(model_dir)
        logger.critical(f'Load model from {model_dir}')
        return

    def _train_model(self, train_loader, valid_loader, **kwargs):
        """Train the model.

        Args:
            train_loader (EasyTPP.DataLoader): data loader for the train set.
            valid_loader (EasyTPP.DataLoader): data loader for the valid set.
        """
        test_loader = kwargs.get('test_loader')
        for i in range(self.runner_config.trainer_config.max_epoch):
            train_metrics = self.run_one_epoch(train_loader, RunnerPhase.TRAIN)

            message = f"[ Epoch {i} (train) ]: train " + MetricsHelper.metrics_dict_to_str(train_metrics)
            logger.info(message)

            self.model_wrapper.write_summary(i, train_metrics, RunnerPhase.TRAIN)

            # evaluate model
            if i % self.runner_config.trainer_config.valid_freq == 0:
                valid_metrics = self.run_one_epoch(valid_loader, RunnerPhase.VALIDATE)

                self.model_wrapper.write_summary(i, valid_metrics, RunnerPhase.VALIDATE)

                message = f"[ Epoch {i} (valid) ]:  valid " + MetricsHelper.metrics_dict_to_str(valid_metrics)
                logger.info(message)

                updated = self.metrics_tracker.update_best("loglike", valid_metrics['loglike'], i)

                message_valid = "current best loglike on valid set is {:.4f} (updated at epoch-{})".format(
                    self.metrics_tracker.current_best['loglike'], self.metrics_tracker.episode_best)

                if updated:
                    message_valid += f", best updated at this epoch"
                    self.model_wrapper.save(self.runner_config.base_config.specs['saved_model_dir'])

                if test_loader is not None:
                    test_metrics = self.run_one_epoch(test_loader, RunnerPhase.VALIDATE)

                    message = f"[ Epoch {i} (test) ]: test " + MetricsHelper.metrics_dict_to_str(test_metrics)
                    logger.info(message)

                logger.critical(message_valid)

        self.model_wrapper.close_summary()

        return

    def _evaluate_model(self, data_loader, **kwargs):
        """Evaluate the model on the valid dataset.

        Args:
            data_loader (EasyTPP.DataLoader): data loader for the valid set

        Returns:
            dict: metrics dict.
        """

        eval_metrics, epoch_pred, epoch_label, epoch_mask = self.run_one_epoch(
            data_loader, RunnerPhase.VALIDATE, return_predictions=True
        )

        self.model_wrapper.write_summary(0, eval_metrics, RunnerPhase.VALIDATE)

        self.model_wrapper.close_summary()

        message = f"Evaluation result: " + MetricsHelper.metrics_dict_to_str(eval_metrics)

        logger.critical(message)

        # Print predictions aligned with ground truth
        if epoch_pred is not None and epoch_label is not None:
            pred_dtime, pred_type = epoch_pred
            label_dtime, label_type = epoch_label

            if pred_dtime is not None and label_dtime is not None:
                self._print_predictions(pred_dtime, pred_type, label_dtime, label_type, epoch_mask)

        return eval_metrics

    def _print_predictions(self, pred_dtime, pred_type, label_dtime, label_type, mask=None):
        """Print predictions aligned with ground truth.

        Args:
            pred_dtime: Predicted delta times, shape [batch, seq_len]
            pred_type: Predicted event types, shape [batch, seq_len]
            label_dtime: Ground truth delta times, shape [batch, seq_len]
            label_type: Ground truth event types, shape [batch, seq_len]
            mask: Valid event mask, shape [batch, seq_len]
        """
        import numpy as np

        print("\n" + "=" * 100)
        print("PREDICTIONS VS GROUND TRUTH (excluding first event per sequence)")
        print("=" * 100)

        num_sequences = pred_dtime.shape[0]
        max_seq_len = pred_dtime.shape[1]

        # Print header
        print(f"\nTotal sequences: {num_sequences}, Max sequence length: {max_seq_len}")
        print("-" * 100)

        # Print detailed results for first few sequences
        num_to_print = min(5, num_sequences)  # Print first 5 sequences

        for seq_idx in range(num_to_print):
            print(f"\n[Sequence {seq_idx + 1}]")
            print(f"{'Event':<8} | {'Pred Type':<10} | {'True Type':<10} | {'Type Match':<12} | {'Pred dTime':<12} | {'True dTime':<12} | {'dTime Error':<12}")
            print("-" * 100)

            # Skip first event (event_idx=0) - start from event_idx=1
            for event_idx in range(1, max_seq_len):
                # Check if this event is valid (not masked/padded)
                if mask is not None and not mask[seq_idx, event_idx]:
                    continue

                p_type = int(pred_type[seq_idx, event_idx])
                t_type = int(label_type[seq_idx, event_idx])
                p_dtime = pred_dtime[seq_idx, event_idx]
                t_dtime = label_dtime[seq_idx, event_idx]

                type_match = "✓" if p_type == t_type else "✗"
                dtime_error = abs(p_dtime - t_dtime)

                print(f"{event_idx + 1:<8} | {p_type:<10} | {t_type:<10} | {type_match:<12} | {p_dtime:<12.4f} | {t_dtime:<12.4f} | {dtime_error:<12.4f}")

        # Print summary statistics (excluding first event per sequence)
        print("\n" + "=" * 100)
        print("SUMMARY STATISTICS (excluding first event per sequence)")
        print("=" * 100)

        # Exclude first event from each sequence
        pred_dtime_excl = pred_dtime[:, 1:]
        pred_type_excl = pred_type[:, 1:]
        label_dtime_excl = label_dtime[:, 1:]
        label_type_excl = label_type[:, 1:]

        if mask is not None:
            mask_excl = mask[:, 1:]
            valid_mask = mask_excl.astype(bool)
            valid_pred_type = pred_type_excl[valid_mask]
            valid_label_type = label_type_excl[valid_mask]
            valid_pred_dtime = pred_dtime_excl[valid_mask]
            valid_label_dtime = label_dtime_excl[valid_mask]
        else:
            valid_pred_type = pred_type_excl.flatten()
            valid_label_type = label_type_excl.flatten()
            valid_pred_dtime = pred_dtime_excl.flatten()
            valid_label_dtime = label_dtime_excl.flatten()

        # Type prediction accuracy
        type_correct = np.sum(valid_pred_type == valid_label_type)
        type_total = len(valid_pred_type)
        type_acc = type_correct / type_total if type_total > 0 else 0

        # Time prediction errors
        dtime_errors = np.abs(valid_pred_dtime - valid_label_dtime)
        rmse = np.sqrt(np.mean(dtime_errors ** 2))
        mae = np.mean(dtime_errors)

        print(f"\nType Prediction:")
        print(f"  - Accuracy: {type_acc:.4f} ({type_correct}/{type_total})")
        print(f"  - Unique predicted types: {np.unique(valid_pred_type).tolist()}")
        print(f"  - Unique true types: {np.unique(valid_label_type).tolist()}")

        print(f"\nTime Prediction:")
        print(f"  - RMSE: {rmse:.4f}")
        print(f"  - MAE: {mae:.4f}")
        print(f"  - Mean predicted dtime: {np.mean(valid_pred_dtime):.4f}")
        print(f"  - Mean true dtime: {np.mean(valid_label_dtime):.4f}")
        print(f"  - Std predicted dtime: {np.std(valid_pred_dtime):.4f}")
        print(f"  - Std true dtime: {np.std(valid_label_dtime):.4f}")

        # Print top 5 dTime errors
        print("\n" + "-" * 100)
        print("TOP 5 LARGEST dTime ERRORS (excluding first event per sequence)")
        print("-" * 100)
        print(f"{'Seq':<6} | {'Event':<8} | {'Pred Type':<10} | {'True Type':<10} | {'Type Match':<12} | {'Pred dTime':<12} | {'True dTime':<12} | {'dTime Error':<12}")
        print("-" * 100)

        # Build list of all errors with their metadata (excluding first event)
        error_list = []
        for seq_idx in range(num_sequences):
            for event_idx in range(1, max_seq_len):
                if mask is not None and not mask[seq_idx, event_idx]:
                    continue
                p_type = int(pred_type[seq_idx, event_idx])
                t_type = int(label_type[seq_idx, event_idx])
                p_dtime = pred_dtime[seq_idx, event_idx]
                t_dtime = label_dtime[seq_idx, event_idx]
                dtime_error = abs(p_dtime - t_dtime)
                error_list.append((dtime_error, seq_idx, event_idx, p_type, t_type, p_dtime, t_dtime))

        # Sort by error (descending) and get top 5
        error_list.sort(key=lambda x: x[0], reverse=True)
        for i, (dtime_error, seq_idx, event_idx, p_type, t_type, p_dtime, t_dtime) in enumerate(error_list[:5]):
            type_match = "✓" if p_type == t_type else "✗"
            print(f"{seq_idx + 1:<6} | {event_idx + 1:<8} | {p_type:<10} | {t_type:<10} | {type_match:<12} | {p_dtime:<12.4f} | {t_dtime:<12.4f} | {dtime_error:<12.4f}")

        print("\n" + "=" * 100)

    def _gen_model(self, data_loader, **kwargs):
        """Generation of the TPP, one-step and multi-step are both supported.
        """

        test_result = self.run_one_epoch(data_loader, RunnerPhase.PREDICT)

        # For the moment we save it to a pkl

        message = f'Save the prediction to pickle file pred.pkl'

        logger.critical(message)

        save_pickle('pred.pkl', test_result)

        return

    def run_one_epoch(self, data_loader, phase, return_predictions=False):
        """Run one complete epoch.

        Args:
            data_loader: data loader object defined in model runner
            phase: enum, [train, dev, test]
            return_predictions: bool, if True, return predictions and labels along with metrics

        Returns:
            a dict of metrics (or tuple of metrics, predictions, labels, mask if return_predictions=True)
        """
        total_loss = 0
        total_num_event = 0
        epoch_label = []
        epoch_pred = []
        epoch_mask = []
        pad_index = self.runner_config.data_config.data_specs.pad_token_id
        metrics_dict = OrderedDict()
        if phase in [RunnerPhase.TRAIN, RunnerPhase.VALIDATE]:
            for batch in data_loader:
                batch_loss, batch_num_event, batch_pred, batch_label, batch_mask = \
                    self.model_wrapper.run_batch(batch, phase=phase)

                total_loss += batch_loss
                total_num_event += batch_num_event
                epoch_pred.append(batch_pred)
                epoch_label.append(batch_label)
                epoch_mask.append(batch_mask)

            avg_loss = total_loss / total_num_event

            metrics_dict.update({'loglike': -avg_loss, 'num_events': total_num_event})

        else:
            for batch in data_loader:
                batch_pred, batch_label = self.model_wrapper.run_batch(batch, phase=phase)
                epoch_pred.append(batch_pred)
                epoch_label.append(batch_label)

        # we need to improve the code here
        # classify batch_output to list
        pred_exists, label_exists = False, False
        concat_epoch_pred, concat_epoch_label, concat_epoch_mask = None, None, None
        if epoch_pred[0][0] is not None:
            concat_epoch_pred = concat_element(epoch_pred, pad_index)
            pred_exists = True
        if len(epoch_label) > 0 and epoch_label[0][0] is not None:
            concat_epoch_label = concat_element(epoch_label, pad_index)
            label_exists = True
            if len(epoch_mask):
                concat_epoch_mask = concat_element(epoch_mask, False)[0]  # retrieve the first element of concat array
                concat_epoch_mask = concat_epoch_mask.astype(bool)

        if pred_exists and label_exists:
            metrics_dict.update(self.metric_functions(concat_epoch_pred, concat_epoch_label, seq_mask=concat_epoch_mask))

        if phase == RunnerPhase.PREDICT:
            metrics_dict.update({'pred': concat_epoch_pred, 'label': concat_epoch_label})

        if return_predictions:
            return metrics_dict, concat_epoch_pred, concat_epoch_label, concat_epoch_mask

        return metrics_dict
