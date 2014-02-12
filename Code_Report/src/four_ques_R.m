function PCA()
    clear;
    imgs = dir('*.pgm');
    
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

function accuracy = four_fold(imgs, fold_no, p, q, k)

    NoImages = size(imgs,1);
    p = 100; q = 100; k = 25;
    
    bit = 0;
    pointer = 0;
    labels = cell(NoImages,1);

    for i=1:NoImages
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
        count = count +1;
        E(:, i) = E(:, i)/norm(E(:, i));
    end
    
    count;
    for i=1:pointer
        for l= 1:k
            FV(i, l) = transpose(E(:, l)) * A(:, i);
            count = count + 1;
        end
    end
    
    count = 0;
    
    %Testing here
    
    right = 0;
    tfinal = 0;
    
    if(fold_no ==0)
        count;
        fold_no =4;
    end
    
    for i=50:50
       
        image = double(imread(imgs(i).name));

        % imgs(i+1).name;
        imgs(i).name;
        b = trans(image);
        b = b - mean;

        % Comput the Feature vector for the testing image.
        testFV = zeros(1, k);
        for j=1:k
            count = count + 1;
            testFV(j) = transpose(E(:, j)) * b;
        end
        
        count;
        
        B=[];
        sumofcols=[];
        count = 0;
        bit=0;
        for j=1:k
            count = count + 1;
            B(:,j) = testFV(j) * E(:,j);
            if(bit==0)
                count = count + 1;
                sumofcols=B(:,j);
                bit=1;
            else
                count = count - 1;
                sumofcols=sumofcols+B(:,j);
            end
            count;
        end
        B = sumofcols;
        B = (B + mean);
        count = 0;
        test_image_fin = reshape(B, [p, q]);
        test = 0;
        imshow(uint8(test_image_fin'));
        
        % Compute k-NN on the testing image.
        dx = 1;
        test = 0;
        min = 10000;

        for j=1:pointer
            count = 0;
            if(j==1)
                count = count + 1;
                min = norm((testFV - FV(j, :)));
            else
                count = count - 1;
                temp = norm((testFV - FV(j, :)));
                if(temp  < min)
                    count;
                    dx = j;
                    min = temp;
                end
                count = 0;
            end
            count;

        end

        if(strcmpi(imgs(i).name(1:7), labels(dx)) ==1)
            count = 0;
            right = right +1;
        end
        count;
        tfinal = tfinal + 1;
    end
    accuracy = double(right/tfinal) * 100;
    
end