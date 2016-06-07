setup() ;

%% Experiment with the res_mnist_fc_bnorm
res_19 = load('/Users/ezeddin/Project/BIK_Project/data/epoch-25-resnet-19-layers.mat');
plain_19 = load('/Users/ezeddin/Project/BIK_Project/data/epoch-25-plain-19-layers.mat');

res_37 = load('/Users/ezeddin/Project/BIK_Project/data/epoch-25-resnet-37-layers.mat');
plain_37 = load('/Users/ezeddin/Project/BIK_Project/data/epoch-25-plain-37-layers.mat');

res_55 = load('/Users/ezeddin/Project/BIK_Project/data/epoch-25-resnet-55-layers.mat');
plain_55 = load('/Users/ezeddin/Project/BIK_Project/data/epoch-25-plain-55-layers.mat');

res_73 = load('/Users/ezeddin/Project/BIK_Project/data/epoch-25-resnet-73-layers.mat');
plain_73 = load('/Users/ezeddin/Project/BIK_Project/data/epoch-25-plain-73-layers.mat');

resnet_str_19 = 'ResNet-19'; plain_str_19 = 'Plain-19';
resnet_str_37 = 'ResNet-37'; plain_str_37 = 'Plain-37';
resnet_str_55 = 'ResNet-55'; plain_str_55 = 'Plain-55';
resnet_str_73 = 'ResNet-73'; plain_str_73 = 'Plain-73';
info_res_19 = res_19.info; info_plain_19 = plain_19.info;
info_res_37 = res_37.info; info_plain_37 = plain_37.info;
info_res_55 = res_55.info; info_plain_55 = plain_55.info;
info_res_73 = res_73.info; info_plain_73 = plain_73.info;

ylimit_res_train = min([max(max(info_res_19.train.error)),max(max(info_res_37.train.error)),max(max(info_res_55.train.error)),max(max(info_res_73.train.error))]);
ylimit_plain_train = min([max(max(info_plain_19.train.error)),max(max(info_plain_37.train.error)),max(max(info_plain_55.train.error)),max(max(info_plain_73.train.error))]);
ylimit_train = min([ylimit_res_train,ylimit_plain_train]);

ylimit_res_val = min([max(max(info_res_19.val.error)),max(max(info_res_37.val.error)),max(max(info_res_55.val.error)),max(max(info_res_73.val.error))]);
ylimit_plain_val = min([max(max(info_plain_19.val.error)),max(max(info_plain_37.val.error)),max(max(info_plain_55.val.error)),max(max(info_plain_73.val.error))]);
ylimit_val = min([ylimit_res_val,ylimit_plain_val]);

ylimit = ylimit_train;
val = true;
stacked_bar = true;


if stacked_bar
    hFig = figure(1);
    set(hFig, 'Position', [500,500,700,400]); clf;
    y = [
        min(info_res_19.val.error(1,:)), min(info_plain_19.val.error(1,:))
        min(info_res_37.val.error(1,:)), min(info_plain_37.val.error(1,:))
        min(info_res_55.val.error(1,:)), min(info_plain_55.val.error(1,:))
        min(info_res_73.val.error(1,:)), min(info_plain_73.val.error(1,:))];
    bar(y);
    str = {'ResNet-19  Plain-19','ResNet-37  Plain-37','ResNet-55  Plain-55', 'ResNet-73  Plain-73'};
    set(gca, 'XTick' , 1:numel(str) , 'XTickLabel' , str )
    ylabel('Validation error') ;
    breakInfo = breakyaxis([0.03 0.888]);

else
    hFig = figure(1);
    set(hFig, 'Position', [500,500,700,400]); clf;
    subplot(1,2,2) ;
    hold on;
    if val
        plot([info_res_19.val.error(1,:); info_res_37.val.error(1,:); info_res_55.val.error(1,:);info_res_73.val.error(1,:)]') ;
        ylabel('Validation error') ;
    else
        plot([info_res_19.train.error(1,:); info_res_37.train.error(1,:); info_res_55.train.error(1,:);info_res_73.train.error(1,:)]') ;
        ylabel('Training error') ;
    end

    xlabel('Epochs'); 
    grid on ;
    h=legend(resnet_str_19,resnet_str_37, resnet_str_55, resnet_str_73) ;
    set(h,'color','none');
    ylim([0 ylimit])
    title('ResNet')

    subplot(1,2,1) ;
    hold on;

    if val
        plot([info_plain_19.val.error(1,:); info_plain_37.val.error(1,:); info_plain_55.val.error(1,:);info_plain_73.val.error(1,:)]') ;
        ylabel('Validation error') ;
    else
        plot([info_plain_19.train.error(1,:); info_plain_37.train.error(1,:); info_plain_55.train.error(1,:);info_plain_73.train.error(1,:)]') ;
        ylabel('Training error') ;
    end
    % ax = gca;
    % ax.ColorOrderIndex = 1;
    xlabel('Epochs');
    grid on ;
    h=legend(plain_str_19,plain_str_37, plain_str_55, plain_str_73) ;
    set(h,'color','none');
    ylim([0 ylimit])
    title('Plain')
end
drawnow ;