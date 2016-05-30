setup() ;
%% Experiment with the cnn_mnist_fc_bnorm

[net_bn, info_bn] = cnn_mnist(...
  'expDir', 'data/cnn-mnist-bnorm', 'batchNormalization', true);


figure(1) ; clf ;
subplot(1,2,1) ;
semilogy(info_bn.val.objective', '+--') ;
xlabel('Training samples [x 10^3]'); ylabel('energy') ;
grid on ;
h=legend('BNORM') ;
set(h,'color','none');
title('CNN objective') ;

subplot(1,2,2) ;
plot(info_bn.val.error', '+--') ;
h=legend('BNORM-val','BNORM-val-5') ;
grid on ;
xlabel('Training samples [x 10^3]'); ylabel('error') ;
set(h,'color','none') ;
title('CNN error') ;
drawnow ;