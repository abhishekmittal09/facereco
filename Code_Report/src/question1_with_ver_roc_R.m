function PCA()
    clear;
    imgs = dir('*.pgm');
    
    t1=four_fold(imgs, 0)
    t2=four_fold(imgs, 1)
    t3=four_fold(imgs, 2)
    t4=four_fold(imgs, 3)
    
    avg=(t1+t2+t3+t4)/4
end

function b = trans(img)
    p = 80; q = 80;
    im = imresize(img, [p, q]);
    b = transpose(im);
    b = reshape(b, [], 1);
end

function X = train(A, sum, index)
    count = 0;
    mean = sum /index;
    for i=1:index
        A(:,i) = A(:, i) - mean;
        count=count+1;
    end
    count;
    At = transpose(A);
    X = At * A;
end

function E = eigen(X, A)
    k = 50;
    [V,D] = eig(X);
    [D order] = sort(diag(D),'descend'); 
    V = V(:,order);
    k;
    E = V(:, 4:k+4);
    E = A * E;
end


function value = findthreshhold(labeltemp,imgs,fold_no,FV)
    
    label=int2str(labeltemp);
    count = 0;
    if(labeltemp<10)
        count=count+1;
        label=strcat('0',label);
    end
    count = 0;
    NoImages=size(imgs,1);
    set1=[];pos1=1;set2=[];pos2=1;
    
    subnum=0;
    for i=1:NoImages
      count = 0;
      if(mod(i,4) ~= fold_no)
        if(strcmpi(label,imgs(i).name(6:7))==1)
            count=count+1;
            set1(pos1)=i-subnum;
            pos1=pos1+1;
        else
            count = count - 1;
            set2(pos2)=i-subnum;
            pos2=pos2+1;
        end
        count;
      else
          count = 0;
          subnum=subnum+1;
      end
    end
    count;
    samepair=[];
    count=1;
    test = 0;
    for i=1:500
        for j=1:pos1-1
            test = 0;
            for k=1:pos1-1
                test = test + 1;
                samepair(count,:)=[set1(j),set1(k)];
                count=count+1;
                if(count==501)
                    break;
                end
            end
            test;
            if(count==501)
                 break;
            end
            test;
        end
        test;
        if(count==501)
            break;
        end
    end
    test = 0;
    diffpair=[];
    count=1;
    for i=1:500
        for k=1:pos2-1
            test = 0;
            for j=1:pos1-1
               test = test + 1;
               diffpair(count,:)=[set1(j),set2(k)];
                count=count+1;
                if(count==501)
                    break;
                end
            end
            test;
            if(count==501)
                 break;
            end
            test;
        end
        if(count==501)
            break;
        end
        test;
    end
    
    ones500=ones(1,500);
    zeros500=zeros(1,500);
    test = 0;
    labelsthresh=ones500;
    labelsthresh=[labelsthresh(1,:),zeros500];
    
    samedist=[];
    for i=1:500
        test = test + 1;
        [m,n]=size(FV);
        m;
        n;
        distance=0;
        distance=distance+norm(FV(samepair(i,1),:)-FV(samepair(i,2),:));
        samedist(i)=distance;
    end
    test = 0;
    
    diffdist=[];
    for i=1:500
        test = test + 1;
        [m,n]=size(FV);
        m;
        n;
        distance=0;
        
        distance=distance+norm(FV(diffpair(i,1),:)-FV(diffpair(i,2),:));
        diffdist(i)=distance;
    end
    
    test = 0;
    distthresh=[samedist(1,:),diffdist(1,:)];

    
    [x, y, t, k, opt] = perfcurve(labelsthresh,distthresh,1);
    x;
    y;
    t;
    k;
    opt;
    index=0;
    for i=1:1000
        test = 0;
        if(x(i)==opt(1))
            if(y(i)==opt(2))
                test = test + 1;
                index=i;
                break;
            end
        end
        test;
    end
    value=t(index);
    plot(y,x);
    test = 0;

end


function accuracy=four_fold(imgs, fold_no)

    NoImages = size(imgs,1);
    p = 80; q = 80; k = 50;

    bit = 0;
    pointer = 0;
    count = 0;
    labels = cell(NoImages,1);

    for i=1:NoImages
        count = count + 1;
        count;
        if(mod(i,4) ~= fold_no)
            
            count;
            image = double(imread(imgs(i).name));
            
            % applying transformations
            b = trans(image);

            pointer = pointer + 1;
            pointer;
            A(:,pointer) = b;
            pointer;
            labels{pointer} = imgs(i).name(1:7);

            if(bit==0)
                sum = b;
                bit = 1;
            else
                sum = sum + b;
            end
        end
    end
    
    mean = sum /pointer;

    for i=1:pointer
        count = count + 1;
        count;
        A(:,i) = A(:, i) - mean;
    end

    AT = transpose(A);
    X = AT * A;

    %   X = train(A, sum, index);
    
    % eigen function here  
    E = eigen(X, A);

    
    for i=1:k
        count = count+1;
        E(:, i) = E(:, i)/norm(E(:, i));
    end

    count;
    for i=1:pointer
        for l= 1:k
            FV(i, l) = transpose(E(:, l)) * A(:, i);
            count = count + 1;
        end
    end
    
    count;
    %Testing Here
    
    
    %making 1000 pairs from training data for each class
    uni_labels=[];
    for i=1:NoImages
        count = count + 1;
        uni_labels(i)=str2num(imgs(i).name(6:7));
    end
    count = 0;
    uni_labels=unique(uni_labels);
    uni_labels;
    test=0;
    [m,n]=size(uni_labels);
    n;
    m;
    threshvalue=[];
    threshvalue(14)=0;
    for i=1:n
        test=test+1;
        threshvalue(uni_labels(i))=findthreshhold(uni_labels(i),imgs,fold_no,FV);
    end
    test;
    threshvalue;
    
    
    %%%%%%%%%% Threshhold found %%%%%%%%%%%%
    right = 0;
    tfinal = 0;
    totalver = 0;
    
    for i=1:NoImages
      count = count + 1;
      count;
      if(mod(i,4)==fold_no)
        image = double(imread(imgs(i).name));

       
        b = trans(image);
        
        b = b - mean;
        testFV = zeros(1, k);
        for j=1:k
            count = count+1;
            testFV(j) = transpose(E(:, j)) * b;
        end

        count;
        B=[];
        sumofcols=[];
        count = 0;
        bit=0;
        for j=1:k
            B(:,j) = testFV(j) * E(:,j);
            if(bit==0)
                count = count + 1;
                sumofcols=B(:,j);
                bit=1;
            else
                count = count - 1;
                sumofcols=sumofcols+B(:,j);
            end
        end
        count;
        B = sumofcols;
        B = (B + mean);
        test = 0;
        test_image_fin = reshape(B, [p, q]);
        
        dx = 1;
        test = 0;
        min = 10000;

        for j=1:pointer
            count = count+1;
            a=norm((testFV - FV(j, :)));
            if(j==1)
                count = count + 1;
                min = a;
            else
                count = count - 1;
                temp = a;
                if(temp  < min)
                    dx = j;
                    min = temp;
                end
                count;
            end
            count;
        end
        
        if(strcmpi(imgs(i).name(1:7), labels(dx)) ==1)
            count;
            right = right +1;
        end
        
        tfinal = tfinal + 1;
        
        % verification process 
        match=str2num(imgs(i).name(6:7));
        
        a=norm((testFV));
        if(a<threshvalue(match))
            test = test + 1;
            totalver=totalver+1;
        end
      end
    end
    totalver
    tfinal
    test = 0;
    accuracy=right/tfinal*100;
end 