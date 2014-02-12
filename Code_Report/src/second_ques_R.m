function second_ques()
    clear;
    imgs = dir('*.png');
    
    t1 = four_fold(imgs, 0)
    t2 = four_fold(imgs, 1)
    t3 = four_fold(imgs, 2)
    t4 = four_fold(imgs, 3)
    
    final_acc = (t1 + t2 + t3 + t4)/4
end

function b = trans(img)
    p = 100; q = 100;
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
    k = 25;
    [V,D] = eig(X);
    [D order] = sort(diag(D),'descend'); 
    V = V(:,order);
    k;
    E = V(:, 4:k+4);
    E = A * E;
end


function accuracy = four_fold(imgs, fold_no)
    
    p = 100; q = 100; k = 25;
    NoImages = size(imgs,1);

    bit = 0;
    pointer = 0;
    count = 0;
    
    labels = cell(NoImages,1);
    count;
    label_of_image=1;
    
    for i=1:NoImages
        count = count+1;
        count;
        if(mod(i,4) ~= fold_no)
            
            count;
            image = double(imread(imgs(i).name));
            
            b = trans(image);

            pointer = pointer + 1;
            pointer;
            A(:,pointer) = b;
            pointer;
            labels{pointer} = int2str(label_of_image);
            count;

            if(bit==0)
                count = count+1;
                sum = b;
                bit = 1;
            else
                count = count-1;
                sum = sum + b;
            end
            count;
        end
        
        if(mod(i,42)==0)
            count;
            label_of_image=label_of_image+1;
        end
    end
    
    mean = sum /pointer;

    count = 0;
    for i=1:pointer
        count=count+1;
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
        count = 0;
        for l= 1:k
            count = count+1;
            FV(i, l) = transpose(E(:, l)) * A(:, i);
        end
        count;
    end
    
    %Testing Start
    
    right = 0;
    tfinal = 0;
    
    if(fold_no ==0)
        count;
        fold_no =4;
    end
    
    for i=fold_no:4:NoImages
        count=count+1;
        count;
        image = double(imread(imgs(i).name));
       % imgs(i+1).name;
        b = trans(image);
        b = b - mean;
        
       
        testFV = zeros(1, k);
        count = 0;
        for j=1:k
            count = count+1;
            testFV(j) = transpose(E(:, j)) * b;
        end
        count;
        
        dx = 1;
        test = 0;
        min = 10000;

        for j=1:pointer
            count = count+1;
            if(j==1)
                count;
                min = norm((testFV - FV(j, :)));
            else
                count;
                temp = norm((testFV - FV(j, :)));
                if(temp  < min)
                    dx = j;
                    min = temp;
                end
                count;
            end
            count;
        end
        label_test=int32(0);
        if(mod(i,42)==0)
            no=int32(i);
            count = count + 1;
            label_test=idivide(no,42);
        else
            count = count - 1;
            no=int32(i);
            label_test=idivide(no,42)+1;
        end
        count;
        
        if(strcmpi(int2str(label_test), labels(dx)) ==1)
            count = count + 1;
            right = right +1;
        end
        count;
        tfinal = tfinal + 1;
    end
    accuracy = double(right/tfinal) * 100;
    
end