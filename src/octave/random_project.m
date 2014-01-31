function [R k] = random_project(d, n, epsilon)
%d: dimension of the data
%n: number of instances
%epsilon: error tolerance

  if ~exist('epsilon', 'var'), epsilon = .1; end;

  k = ceil(log(n/epsilon^2)) + 1;
  R = zeros(d,k);
  rnd = rand(d,k);

  R(rnd < 1/6) = -sqrt(3);
  R(rnd > 5/6) = sqrt(3);
end
