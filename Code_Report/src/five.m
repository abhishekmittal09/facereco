function PCA()
    clear;
    imgs = dir('*.bmp');
    p = 40; q = 40; k = 50;
    a1 = four_fold(imgs, 0, p, q, k)
    a2 = four_fold(imgs, 1, p, q, k)
    a3 = four_fold(imgs, 2, p, q, k)
    a4 = four_fold(imgs, 3, p, q, k)
    
    mean_accuracy = (a1 + a2 + a3 + a4)/4
end

function b = trans(img, p, q)
    % do a resize here.
    im = imresize(img, [p, q]);
    % Convert the given image to column vector.
    b = transpose(im);
    b = reshape(b, [], 1);
end

function E = eigen(X, k, A)
    % Compute the eigen vector, eigen values of the AtA
    [V,D] = eig(X);
    [D order] = sort(diag(D),'descend'); 
    V = V(:,order);

    % select top k eigen vectors.
    E = V(:, 4:k+4);

    % Compute the eigen faces.
    E = A * E;
end
  
function accuracy = four_fold(imgs, fold_no, p, q, k)

    NumImgs = size(imgs,1);

    flag = 0;
    index = 0;
    labels = cell(NumImgs,1);
    count = 0;

    for i=1:NumImgs
        count=count+1;
        if(mod(i,4) ~= fold_no)
            
            imgs(i).name;
            image = double(imread(imgs(i).name));
            
            % applying transformations
            b = trans(image, p, q);

            index = index + 1;
            A(:,index) = b;
            labels{index} = imgs(i).name(1:9);

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
        count=count+1;
        A(:,i) = A(:, i) - mean;
    end

    At = transpose(A);
    X = At * A;

    E = eigen(X, k, A);

    % Normalise the eigen faces.
    for i=1:k
        E(:, i) = E(:, i)/norm(E(:, i));
    end

    % Compute the Feature vector for each class.
    for i=1:index
        count=count+1;
        for l= 1:k
            FV(i, l) = transpose(E(:, l)) * A(:, i);
        end
    end
    
    %%%%%%%%%% Training Done %%%%%%%%%%%%%%%
    
    correct = 0;
    total = 0;
    
    if(fold_no ==0)
        fold_no =4;
    end
    
    for i=fold_no:4:NumImgs
        count=count+1;
        imgs(i).name;
        image = double(imread(imgs(i).name));

        b = trans(image, p, q);

        % Subtract the mean image.
        b = b - mean;

        % Comput the Feature vector for the testing image.
        testFV = zeros(1, k);
        for j=1:k
            testFV(j) = transpose(E(:, j)) * b;
        end

        % Compute k-NN on the testing image.
        mindx = 1;
        min = 10000;

        for j=1:index
            count=count+1;
            if(j==1)
                min = norm((testFV - FV(j, :)));
            else
                temp = norm((testFV - FV(j, :)));
                if(temp  < min)
                    mindx = j;
                    min = temp;
                end
            end

        end

        if(strcmpi(imgs(i).name(1:9), labels(mindx)) ==1)
            correct = correct +1;
        end

        total = total + 1;
    end
    accuracy = double(correct/total) * 100;
    
end

%mean_image = reshape(mean, 168, 192);
%mean_image = transpose(mean_image); 