function PCA()
    clear;
    imgs = dir('*.pgm');
    p = 80; q = 80; k = 50;
    a1=four_fold(imgs, 0, p, q, k)
    a2=four_fold(imgs, 1, p, q, k)
    a3=four_fold(imgs, 2, p, q, k)
    a4=four_fold(imgs, 3, p, q, k)
    avg=(a1+a2+a3+a4)/4
end

function value=findthreshhold(labeltemp,imgs,fold_no,FV)
    label=int2str(labeltemp);
    if(labeltemp<10)
        label=strcat('0',label);
    end
    NumImgs=size(imgs,1);
    set1=[];pos1=1;set2=[];pos2=1;
    
    subnum=0;
    for i=1:NumImgs
      if(mod(i,4) ~= fold_no)
        if(strcmpi(label,imgs(i).name(6:7))==1)
            set1(pos1)=i-subnum;
            pos1=pos1+1;
        else
            set2(pos2)=i-subnum;
            pos2=pos2+1;
        end
      else
          subnum=subnum+1;
      end
    end
    samepair=[];
    count=1;
    for i=1:500
        for j=1:pos1-1
            for k=1:pos1-1
                samepair(count,:)=[set1(j),set1(k)];
                count=count+1;
                if(count==501)
                    break;
                end
            end
            if(count==501)
                 break;
            end
        end
        if(count==501)
            break;
        end
    end
    diffpair=[];
    count=1;
    for i=1:500
        for k=1:pos2-1
            for j=1:pos1-1
               diffpair(count,:)=[set1(j),set2(k)];
                count=count+1;
                if(count==501)
                    break;
                end
            end
            if(count==501)
                 break;
            end
        end
        if(count==501)
            break;
        end
    end
    
    ones500=ones(1,500);
    zeros500=zeros(1,500);
    labelsthresh=ones500;
    labelsthresh=[labelsthresh(1,:),zeros500];
    
    samedist=[];
    for i=1:500
        [m,n]=size(FV);
        distance=0;
        %for j=1:n
             %distance=distance+(FV(samepair(i,1),j)-FV(samepair(i,2),j))*(FV(samepair(i,1),j)-FV(samepair(i,2),j));
        distance=distance+norm(FV(samepair(i,1),:)-FV(samepair(i,2),:));
        %end
        samedist(i)=distance;
    end
    
    diffdist=[];
    for i=1:500
        [m,n]=size(FV);
        distance=0;
        %for j=1:n
             %distance=distance+(FV(diffpair(i,1),j)-FV(diffpair(i,2),j))*(FV(diffpair(i,1),j)-FV(diffpair(i,2),j));
        %end
        distance=distance+norm(FV(diffpair(i,1),:)-FV(diffpair(i,2),:));
        diffdist(i)=distance;
    end
    
    distthresh=[samedist(1,:),diffdist(1,:)];

    
    [x, y, t, k, opt] = perfcurve(labelsthresh,distthresh,1);
    index=0;
    for i=1:1000
        if(x(i)==opt(1))
            if(y(i)==opt(2))
                index=i;
                break;
            end
        end
    end
    value=t(index);
    plot(y,x);

end


function accuracy=four_fold(imgs, fold_no, p, q, k)

    NumImgs = size(imgs,1);

    flag = 0;
    index = 0;
    labels = cell(NumImgs,1);

    for i=1:NumImgs
        if(mod(i,4) ~= fold_no)
            if(imgs(i).name(1)=='.')
                imgs(i).name=imgs(i).name(3:end);
            elseif(imgs(i).name(1)=='_')
                imgs(i).name=imgs(i).name(2:end);
            else
                imgs(i).name=imgs(i).name(1:end);
            end
            imgs(i).name;
            image = double(imread(imgs(i).name));
            % do a resize here.
            image = imresize(image, [p, q]);

            % Convert the given image to column vector.
            b = transpose(image);
            b = reshape(b, [], 1);

            index = index + 1;
            A(:,index) = b;
            labels{index} = imgs(i).name(1:7);

            if(flag==0)
                sum = b;
                flag = 1;
            else
                sum = sum + b;
            end
        end
    end
    
    % Find the mean image.
    mean = sum /index;

    % Subtract the mean image from all images.
    for i=1:index
        A(:,i) = A(:, i) - mean;
    end

    At = transpose(A);
    X = At * A;

    % Compute the eigen vector, eigen values of the AtA
    [V,D] = eig(X);
    [D order] = sort(diag(D),'descend');
    V = V(:,order);

    % select top k eigen vectors.
    E = V(:, 4:k+4);

    % Compute the eigen faces.
    E = A * E;

    % Normalise the eigen faces.
    for i=1:k
        E(:, i) = E(:, i)/norm(E(:, i));
    end

    % Compute the Feature vector for each class.
    for i=1:index
        for l= 1:k
            FV(i, l) = transpose(E(:, l)) * A(:, i);
        end
    end
    
    %%%%%%%%%% Training Done %%%%%%%%%%%%%%%
    
    
    %making 1000 pairs from training data for each class
    uni_labels=[];
    for i=1:NumImgs
        uni_labels(i)=str2num(imgs(i).name(6:7));
    end
    uni_labels=unique(uni_labels);
    uni_labels;
    [m,n]=size(uni_labels);
    n;
    threshvalue=[];
    threshvalue(14)=0;
    for i=1:n
        threshvalue(uni_labels(i))=findthreshhold(uni_labels(i),imgs,fold_no,FV);
    end
    threshvalue;
    
    
    %%%%%%%%%% Threshhold found %%%%%%%%%%%%
    correct = 0;
    total = 0;
    totalver = 0;
    correct=0;
    total=0;
    for i=1:NumImgs
      if(mod(i,4)==fold_no)
        image = double(imread(imgs(i).name));

        % do a resize here.
        image = imresize(image, [p, q]);
        %image = rgb2gray(image);   %for colored images
        % Convert the image to column matrix/
        b = transpose(image);
        b = reshape(b, [], 1);

        % Subtract the mean image.
        b = b - mean;

        % Comput the Feature vector for the testing image.
        testFV = zeros(1, k);
        for j=1:k
            testFV(j) = transpose(E(:, j)) * b;
        end

        B=[];
        sumofcols=[];
        flag=0;
        for j=1:k
            B(:,j) = testFV(j) * E(:,j);
            if(flag==0)
                sumofcols=B(:,j);
                flag=1;
            else
                sumofcols=sumofcols+B(:,j);
            end
        end
        B = sumofcols;
        B = (B + mean);
        test_image_fin = reshape(B, [p, q]);
        %imshow(uint8(test_image_fin'));
        
        % Compute k-NN on the testing image.
        mindx = 1;
        min = 10000;

        for j=1:index
            a=norm((testFV - FV(j, :)));
            if(j==1)
                min = a;
            else
                temp = a;
                if(temp  < min)
                    mindx = j;
                    min = temp;
                end
            end
        end
        
        if(strcmpi(imgs(i).name(1:7), labels(mindx)) ==1)
            correct = correct +1;
        end
        
        total = total + 1;
        
        %%%%%%%%% verification process %%%%%%%%%
        match=str2num(imgs(i).name(6:7));
        
        a=norm((testFV));
        if(a<threshvalue(match))
            totalver=totalver+1;
        end
      end
    end
    totalver
    total
    accuracy=correct/total*100;
end

%mean_image = reshape(mean, 168, 192);
%mean_image = transpose(mean_image); 