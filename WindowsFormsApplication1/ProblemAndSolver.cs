using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using System.Drawing;
using System.Diagnostics;


namespace TSP
{

    class ProblemAndSolver
    {

        private class TSPSolution
        {
            /// <summary>
            /// we use the representation [cityB,cityA,cityC] 
            /// to mean that cityB is the first city in the solution, cityA is the second, cityC is the third 
            /// and the edge from cityC to cityB is the final edge in the path.  
            /// You are, of course, free to use a different representation if it would be more convenient or efficient 
            /// for your data structure(s) and search algorithm. 
            /// </summary>
            public ArrayList Route;

            /// <summary>
            /// constructor
            /// </summary>
            /// <param name="iroute">a (hopefully) valid tour</param>
            public TSPSolution(ArrayList iroute)
            {
                Route = new ArrayList(iroute);
            }

            /// <summary>
            /// Compute the cost of the current route.  
            /// Note: This does not check that the route is complete.
            /// It assumes that the route passes from the last city back to the first city. 
            /// </summary>
            /// <returns></returns>
            public double costOfRoute()
            {
                // go through each edge in the route and add up the cost. 
                int x;
                City here;
                double cost = 0D;

                for (x = 0; x < Route.Count - 1; x++)
                {
                    here = Route[x] as City;
                    cost += here.costToGetTo(Route[x + 1] as City);
                }

                // go from the last city to the first. 
                here = Route[Route.Count - 1] as City;
                cost += here.costToGetTo(Route[0] as City);
                return cost;
            }
        }

        #region Private members 

        /// <summary>
        /// Default number of cities (unused -- to set defaults, change the values in the GUI form)
        /// </summary>
        // (This is no longer used -- to set default values, edit the form directly.  Open Form1.cs,
        // click on the Problem Size text box, go to the Properties window (lower right corner), 
        // and change the "Text" value.)
        private const int DEFAULT_SIZE = 25;

        /// <summary>
        /// Default time limit (unused -- to set defaults, change the values in the GUI form)
        /// </summary>
        // (This is no longer used -- to set default values, edit the form directly.  Open Form1.cs,
        // click on the Time text box, go to the Properties window (lower right corner), 
        // and change the "Text" value.)
        private const int TIME_LIMIT = 60;        //in seconds

        private const int CITY_ICON_SIZE = 5;


        // For normal and hard modes:
        // hard mode only
        private const double FRACTION_OF_PATHS_TO_REMOVE = 0.20;

        /// <summary>
        /// the cities in the current problem.
        /// </summary>
        private City[] Cities;
        /// <summary>
        /// a route through the current problem, useful as a temporary variable. 
        /// </summary>
        private ArrayList Route;
        /// <summary>
        /// best solution so far. 
        /// </summary>
        private TSPSolution bssf;

        private int[,] routeFrequencies;

        /// <summary>
        /// how to color various things. 
        /// </summary>
        private Brush cityBrushStartStyle;
        private Brush cityBrushStyle;
        private Pen routePenStyle;


        /// <summary>
        /// keep track of the seed value so that the same sequence of problems can be 
        /// regenerated next time the generator is run. 
        /// </summary>
        private int _seed;
        /// <summary>
        /// number of cities to include in a problem. 
        /// </summary>
        private int _size;

        /// <summary>
        /// Difficulty level
        /// </summary>
        private HardMode.Modes _mode;

        /// <summary>
        /// random number generator. 
        /// </summary>
        private Random rnd;

        /// <summary>
        /// time limit in milliseconds for state space search
        /// can be used by any solver method to truncate the search and return the BSSF
        /// </summary>
        private int time_limit;
        #endregion

        #region Public members

        /// <summary>
        /// These three constants are used for convenience/clarity in populating and accessing the results array that is passed back to the calling Form
        /// </summary>
        public const int COST = 0;           
        public const int TIME = 1;
        public const int COUNT = 2;
        
        public int Size
        {
            get { return _size; }
        }

        public int Seed
        {
            get { return _seed; }
        }
        #endregion

        #region Constructors
        public ProblemAndSolver()
        {
            this._seed = 1; 
            rnd = new Random(1);
            this._size = DEFAULT_SIZE;
            this.time_limit = TIME_LIMIT * 1000;                  // TIME_LIMIT is in seconds, but timer wants it in milliseconds

            this.resetData();
        }

        public ProblemAndSolver(int seed)
        {
            this._seed = seed;
            rnd = new Random(seed);
            this._size = DEFAULT_SIZE;
            this.time_limit = TIME_LIMIT * 1000;                  // TIME_LIMIT is in seconds, but timer wants it in milliseconds

            this.resetData();
        }

        public ProblemAndSolver(int seed, int size)
        {
            this._seed = seed;
            this._size = size;
            rnd = new Random(seed);
            this.time_limit = TIME_LIMIT * 1000;                        // TIME_LIMIT is in seconds, but timer wants it in milliseconds

            this.resetData();
        }
        public ProblemAndSolver(int seed, int size, int time)
        {
            this._seed = seed;
            this._size = size;
            rnd = new Random(seed);
            this.time_limit = time*1000;                        // time is entered in the GUI in seconds, but timer wants it in milliseconds

            this.resetData();
        }
        #endregion

        #region Private Methods

        /// <summary>
        /// Reset the problem instance.
        /// </summary>
        private void resetData()
        {

            Cities = new City[_size];
            Route = new ArrayList(_size);
            bssf = null;

            if (_mode == HardMode.Modes.Easy)
            {
                for (int i = 0; i < _size; i++)
                    Cities[i] = new City(rnd.NextDouble(), rnd.NextDouble());
            }
            else // Medium and hard
            {
                for (int i = 0; i < _size; i++)
                    Cities[i] = new City(rnd.NextDouble(), rnd.NextDouble(), rnd.NextDouble() * City.MAX_ELEVATION);
            }

            HardMode mm = new HardMode(this._mode, this.rnd, Cities);
            if (_mode == HardMode.Modes.Hard)
            {
                int edgesToRemove = (int)(_size * FRACTION_OF_PATHS_TO_REMOVE);
                mm.removePaths(edgesToRemove);
            }
            City.setModeManager(mm);

            cityBrushStyle = new SolidBrush(Color.Black);
            cityBrushStartStyle = new SolidBrush(Color.Red);
            routePenStyle = new Pen(Color.Blue,1);
            routePenStyle.DashStyle = System.Drawing.Drawing2D.DashStyle.Solid;
        }

        #endregion

        #region Public Methods

        /// <summary>
        /// make a new problem with the given size.
        /// </summary>
        /// <param name="size">number of cities</param>
        public void GenerateProblem(int size, HardMode.Modes mode)
        {
            this._size = size;
            this._mode = mode;
            resetData();
        }

        /// <summary>
        /// make a new problem with the given size, now including timelimit paremeter that was added to form.
        /// </summary>
        /// <param name="size">number of cities</param>
        public void GenerateProblem(int size, HardMode.Modes mode, int timelimit)
        {
            this._size = size;
            this._mode = mode;
            this.time_limit = timelimit*1000;                                   //convert seconds to milliseconds
            resetData();
        }

        /// <summary>
        /// return a copy of the cities in this problem. 
        /// </summary>
        /// <returns>array of cities</returns>
        public City[] GetCities()
        {
            City[] retCities = new City[Cities.Length];
            Array.Copy(Cities, retCities, Cities.Length);
            return retCities;
        }

        /// <summary>
        /// draw the cities in the problem.  if the bssf member is defined, then
        /// draw that too. 
        /// </summary>
        /// <param name="g">where to draw the stuff</param>
        public void Draw(Graphics g)
        {
            float width  = g.VisibleClipBounds.Width-45F;
            float height = g.VisibleClipBounds.Height-45F;
            Font labelFont = new Font("Arial", 10);

            // Draw lines
            if (bssf != null)
            {
                // make a list of points. 
                Point[] ps = new Point[bssf.Route.Count];
                int index = 0;
                foreach (City c in bssf.Route)
                {
                    if (index < bssf.Route.Count -1)
                        g.DrawString(" " + index +"("+c.costToGetTo(bssf.Route[index+1]as City)+")", labelFont, cityBrushStartStyle, new PointF((float)c.X * width + 3F, (float)c.Y * height));
                    else 
                        g.DrawString(" " + index +"("+c.costToGetTo(bssf.Route[0]as City)+")", labelFont, cityBrushStartStyle, new PointF((float)c.X * width + 3F, (float)c.Y * height));
                    ps[index++] = new Point((int)(c.X * width) + CITY_ICON_SIZE / 2, (int)(c.Y * height) + CITY_ICON_SIZE / 2);
                }

                if (ps.Length > 0)
                {
                    g.DrawLines(routePenStyle, ps);
                    g.FillEllipse(cityBrushStartStyle, (float)Cities[0].X * width - 1, (float)Cities[0].Y * height - 1, CITY_ICON_SIZE + 2, CITY_ICON_SIZE + 2);
                }

                // draw the last line. 
                g.DrawLine(routePenStyle, ps[0], ps[ps.Length - 1]);
            }

            // Draw city dots
            foreach (City c in Cities)
            {
                g.FillEllipse(cityBrushStyle, (float)c.X * width, (float)c.Y * height, CITY_ICON_SIZE, CITY_ICON_SIZE);
            }

        }

        /// <summary>
        ///  return the cost of the best solution so far. 
        /// </summary>
        /// <returns></returns>
        public double costOfBssf()
        {
            if (bssf != null)
                return (bssf.costOfRoute());
            else
                return -1D; 
        }

        /// <summary>
        /// This is the entry point for the default solver
        /// which just finds a valid random tour 
        /// </summary>
        /// <returns>results array for GUI that contains three ints: cost of solution, time spent to find solution, number of solutions found during search (not counting initial BSSF estimate)</returns>
        public string[] defaultSolveProblem()
        {
            string[] results = new string[3];
            Stopwatch timer = new Stopwatch();
            int i, swap, temp, count = 0;
            int[] perm = new int[Cities.Length];
            Route = new ArrayList();
            Random rnd = new Random();

            timer.Start();

            do
            {
                for (i = 0; i < perm.Length; i++)                                 // create a random permutation template
                    perm[i] = i;
                for (i = 0; i < perm.Length; i++)
                {
                    swap = i;
                    while (swap == i)
                        swap = rnd.Next(0, Cities.Length);
                    temp = perm[i];
                    perm[i] = perm[swap];
                    perm[swap] = temp;
                }
                Route.Clear();
                for (i = 0; i < Cities.Length; i++)                            // Now build the route using the random permutation 
                {
                    Route.Add(Cities[perm[i]]);
                }
                bssf = new TSPSolution(Route);
                count++;
            } while (costOfBssf() == double.PositiveInfinity);                // until a valid route is found

            timer.Stop();

            results[COST] = costOfBssf().ToString();                          // load results array
            results[TIME] = timer.Elapsed.ToString();
            results[COUNT] = count.ToString();

            return results;
        }
        

        /// <summary>
        /// performs a Branch and Bound search of the state space of partial tours
        /// stops when time limit expires and uses BSSF as solution
        /// </summary>
        /// <returns>results array for GUI that contains three ints: cost of solution, time spent to find solution, number of solutions found during search (not counting initial BSSF estimate)</returns>
        public string[] bBSolveProblem()
        {
            string[] results = new string[3];
            Stopwatch timer = new Stopwatch();

            // Sets the initial bssf
            GreedySolution();

            timer.Start();
            
            int count = BBSolution(timer);

            timer.Stop();
            results[COST] = costOfBssf().ToString();
            results[TIME] = timer.Elapsed.ToString();
            results[COUNT] = count.ToString();

            return results;
        }

        public int BBSolution(Stopwatch timer)
        {
            // Stats
            int maxStates = 0;
            int totalStates = 0;
            int prunedStates = 0;

            int count = 0;
            PriorityQueue q = new PriorityQueue();
            SubProblem initialSubProblem = new SubProblem(
                this.CalculateCostMatrix(),
                this.Cities.Length,
                0,
                new List<int>()
            );
            initialSubProblem.path.Add(0);
            q.Add(initialSubProblem);
            
            while (!q.Empty() && timer.ElapsedMilliseconds < this.time_limit)
            {
                // Update statistics
                if (q.Size() > maxStates)
                {
                    maxStates = q.Size();
                }
                ++totalStates;

                SubProblem subProblem = q.Pop();
                

                if (subProblem.path.Count == this.Cities.Length &&
                    subProblem.path[subProblem.GetCurrentCityIndex()] != 0)
                {
                    // The subproblem is a valid solution!
                    this.bssf = new TSPSolution(PathToRoute(subProblem.path));
                    count++;
                }
                else
                {
                    if (subProblem.HasMorePaths())
                    {
                        for (int col = 0; col < this.Cities.Length; ++col)
                        {
                            if (subProblem.reducedMatrix[subProblem.GetCurrentCityIndex(), col] != double.PositiveInfinity)
                            {
                                if (subProblem.reducedMatrix[subProblem.GetCurrentCityIndex(), col] + subProblem.lowerBound < this.costOfBssf())
                                {
                                    List<int> pathClone = new List<int>(subProblem.path);
                                    SubProblem newProblem = new SubProblem(
                                        (double[,])subProblem.reducedMatrix.Clone(),
                                        this.Cities.Length,
                                        subProblem.lowerBound,
                                        pathClone
                                    );
                                    newProblem.path.Add(col);
                                    newProblem.BlockPaths(
                                        subProblem.GetCurrentCityIndex(),
                                        newProblem.GetCurrentCityIndex()
                                    );
                                    newProblem.ReduceCostMatrix();
                                    q.Add(newProblem);
                                }
                                else
                                {
                                    ++prunedStates;
                                }
                            }
                        }
                    }
                }
            }

            // Print stats
            Console.WriteLine("Max # of states: {0}", maxStates);
            Console.WriteLine("Total # number of states: {0}", totalStates);
            Console.WriteLine("# of states pruned: {0}", prunedStates);

            return count;
        }

        // Converts the path in a subproblem to a route that can be used
        // in a TSPSolution.
        public ArrayList PathToRoute(List<int> path)
        {
            ArrayList route = new ArrayList();

            for (int i = 0; i < path.Count; ++i)
            {
                route.Add(this.Cities[path[i]]);
            }

            return route;
        }

        public double[,] CalculateCostMatrix()
        {
            double[,] costMatrix = new double[this.Cities.Length, this.Cities.Length];

            for (int i = 0; i < this.Cities.Length; ++i)
            {
                for (int j = 0; j < this.Cities.Length; ++j) {
                    costMatrix[i, j] = this.Cities[i].costToGetTo(this.Cities[j]);
                }
                costMatrix[i, i] = double.PositiveInfinity;
            }

            return costMatrix;
        }

        // Updates a matrix after adding a new city to the path
        public void BlockPaths(double[,] matrix, int i, int j)
        {
            for (int index = 0; index < this.Cities.Length; ++index)
            {
                matrix[i, index] = double.PositiveInfinity;
                matrix[index, j] = double.PositiveInfinity;
            }
            matrix[j, i] = double.PositiveInfinity;
        }

        public bool HasMorePaths(double[,] matrix, int row)
        {
            for (int i = 0; i < this.Cities.Length; ++i)
            {
                if (matrix[row, i] != double.PositiveInfinity)
                {
                    return true;
                }
            }
            return false;
        }

        // Used for the greedy algorithm to get a starting matrix
        public double[,] GetStartMatrix(double[,] costMatrix, int startIndex)
        {
            double[,] matrix = new double[this.Cities.Length, this.Cities.Length];

            for (int i = 0; i < this.Cities.Length; ++i)
            {
                for (int j = 0; j < this.Cities.Length; ++j)
                {
                    if (j == startIndex)
                    {
                        matrix[i, j] = double.PositiveInfinity;
                    }
                    else
                    {
                        matrix[i, j] = costMatrix[i, j];
                    }
                }
            }

            return matrix;
        }

        // Gets the index of the smallest thing on a row
        public int GetMinIndex(double[,] matrix, int row)
        {
            int minIndex = -1;
            double minValue = double.PositiveInfinity;
            for (int col = 0; col < this.Cities.Length; ++col)
            {
                if (matrix[row, col] < minValue)
                {
                    minIndex = col;
                    minValue = matrix[row, col];
                }
            }
            return minIndex;
        }


        /////////////////////////////////////////////////////////////////////////////////////////////
        // These additional solver methods will be implemented as part of the group project.
        ////////////////////////////////////////////////////////////////////////////////////////////

        public int GreedySolution()
        {
            int count = 0;
            
            double[,] costMatrix = this.CalculateCostMatrix();
            for (int startIndex = 0; startIndex < this.Cities.Length; ++startIndex)
            {
                List<int> path = new List<int>();
                path.Add(startIndex);

                int currentRow = startIndex;
                double cost = 0;
                double[,] workingMatrix = GetStartMatrix(costMatrix, startIndex);

                while (this.HasMorePaths(workingMatrix, currentRow))
                {
                    int minIndex = this.GetMinIndex(workingMatrix, currentRow);
                    if (minIndex != -1)
                    {
                        path.Add(minIndex);
                        cost += workingMatrix[currentRow, minIndex];
                        BlockPaths(workingMatrix, currentRow, minIndex);
                        currentRow = minIndex;
                        
                        if (path.Count == this.Cities.Length)
                        {
                            count++;
                            if (this.bssf == null || cost + costMatrix[currentRow, startIndex] < this.bssf.costOfRoute())
                            {
                                bssf = new TSPSolution(this.PathToRoute(path));
                                break;
                            }
                        }
                    }
                }
            }

            return count;
        }

        /// <summary>
        /// finds the greedy tour starting from each city and keeps the best (valid) one
        /// </summary>
        /// <returns>results array for GUI that contains three ints: cost of solution, time spent to find solution, number of solutions found during search (not counting initial BSSF estimate)</returns>
        public string[] greedySolveProblem()
        {
            string[] results = new string[3];
            Stopwatch timer = new Stopwatch();
            
            timer.Start();

            int count = GreedySolution();

            timer.Stop();

            results[COST] = costOfBssf().ToString();    // load results into array here, replacing these dummy values
            results[TIME] = timer.Elapsed.ToString();
            results[COUNT] = count.ToString();

            return results;
        }

        public int greedyAntSolution()
        {
            routeFrequencies = new int[Cities.Length, Cities.Length];
            for(int i = 0; i < Cities.Length; i++)
            {
                for(int j = 0; j < Cities.Length; j++)
                {
                    routeFrequencies[i,j] = 0;
                }
            }
            int count = 0;
            
            double[,] costMatrix = this.CalculateCostMatrix();
            for (int startIndex = 0; startIndex < this.Cities.Length; ++startIndex)
            {
                List<int> path = new List<int>();
                path.Add(startIndex);

                int currentRow = startIndex;
                double cost = 0;
                double[,] workingMatrix = GetStartMatrix(costMatrix, startIndex);

                while (this.HasMorePaths(workingMatrix, currentRow))
                {
                    int minIndex = this.GetMinIndex(workingMatrix, currentRow);
                    if (minIndex != -1)
                    {
                        routeFrequencies[currentRow, minIndex]++;
                        path.Add(minIndex);
                        cost += workingMatrix[currentRow, minIndex];
                        BlockPaths(workingMatrix, currentRow, minIndex);
                        currentRow = minIndex;
                        
                        if (path.Count == this.Cities.Length)
                        {
                            count++;
                            if (this.bssf == null || cost + costMatrix[currentRow, startIndex] < this.bssf.costOfRoute())
                            {
                                bssf = new TSPSolution(this.PathToRoute(path));
                                break;
                            }
                        }
                    }
                }
            }
            return count;
        }

        

        public string[] fancySolveProblem()
        {
            string[] results = new string[3];
            Stopwatch timer = new Stopwatch();

            timer.Start();

            int count = greedyAntSolution();
            double bestCost = costOfBssf();
            for(int start = 0; start < Cities.Length; start++)
            {
                double routeCost = 0;
                ArrayList route = new ArrayList();
                bool routeFound = false;
                int from = start;
                while(!routeFound)
                {
                    int popValue = 0;
                    int popIndex = -1;
                    double shortestValue = double.PositiveInfinity;
                    int shortestIndex = -1;
                    for(int to = 0; to < Cities.Length; to++)
                    {
                        if(!(route.Contains(Cities[to])))
                        {
                            if(Cities[from].costToGetTo(Cities[to]) < shortestValue)
                            {
                                shortestValue = Cities[from].costToGetTo(Cities[to]);
                                shortestIndex = to;
                            }
                            if(routeFrequencies[from, to] >= popValue)
                            {
                                popValue = routeFrequencies[from, to];
                                popIndex = to;
                            }
                        }
                    }
                    if (shortestValue * 1.5 < Cities[from].costToGetTo(Cities[popIndex]))
                    {
                        routeCost += Cities[from].costToGetTo(Cities[shortestIndex]);
                        if(routeCost > bestCost)
                        {
                            break;
                        }
                        route.Add(Cities[shortestIndex]);
                        from = shortestIndex;
                    }
                    else
                    {
                        routeCost += Cities[from].costToGetTo(Cities[popIndex]);
                        if (routeCost > bestCost)
                        {
                            break;
                        }
                        route.Add(Cities[popIndex]);
                        from = popIndex;
                    }
                    if (route.Count == Cities.Length)
                    {
                        routeFound = true;
                    }
                }
                double toStart = Cities[from].costToGetTo(Cities[start]);
                if(toStart != double.PositiveInfinity)
                {
                    routeCost += toStart; 
                    if(routeCost < bestCost)
                    {
                        bssf = new TSPSolution(route);
                        bestCost = costOfBssf();
                        //break
                    }
                }
            }


            timer.Stop();

            results[COST] = costOfBssf().ToString();                          // load results array
            results[TIME] = timer.Elapsed.ToString();
            results[COUNT] = count.ToString();

            return results;
        }
        #endregion
    }



    public class PriorityQueue
    {
        private List<SubProblem> queue;

        public PriorityQueue()
        {
            this.queue = new List<SubProblem>();
        }

        public int Size()
        {
            return this.queue.Count;
        }

        public void MakeQueue(List<SubProblem> subProblems)
        {
            for (int i = 0; i < subProblems.Count; ++i)
            {
                this.Add(subProblems[i]);
            }
        }

        public void Add(SubProblem subProblem)
        {
            this.queue.Add(subProblem);
            this.BubbleUp(this.queue.Count - 1);
        }

        public bool Empty()
        {
            return this.queue.Count == 0;
        }

        public SubProblem Pop()
        {
            SubProblem minimum = this.queue[0];
            this.Swap(0, this.queue.Count - 1);
            this.queue.RemoveAt(queue.Count - 1);

            int i = 0;
            while (this.GetLeftChildIndex(i) < this.queue.Count)
            {
                int leftChildIndex = this.GetLeftChildIndex(i);
                int rightChildIndex = this.GetRightChildIndex(i);
                int minChildIndex;
                if (rightChildIndex >= this.queue.Count)
                {
                    minChildIndex = leftChildIndex;
                }
                else if (this.GetHeuristic(this.queue[leftChildIndex]) < this.GetHeuristic(this.queue[rightChildIndex]))
                {
                    minChildIndex = leftChildIndex;
                }
                else
                {
                    minChildIndex = rightChildIndex;
                }

                if (this.GetHeuristic(this.queue[minChildIndex]) < this.GetHeuristic(this.queue[i]))
                {
                    this.Swap(i, minChildIndex);
                    i = minChildIndex;
                }
                else
                {
                    break;
                }
            }

            return minimum;
        }


        private void Swap(int xi, int yi)
        {
            SubProblem temp = this.queue[xi];
            this.queue[xi] = this.queue[yi];
            this.queue[yi] = temp;
        }
         
        public void BubbleUp(int i)
        {
            int parentIndex = this.GetParentIndex(i);
            while (this.GetHeuristic(this.queue[i]) < this.GetHeuristic(this.queue[parentIndex]))
            {
                this.Swap(i, parentIndex);

                // Update parent indices
                i = parentIndex;
                parentIndex = this.GetParentIndex(i);
            }
        }

        public double GetHeuristic(SubProblem subProblem)
        {
            return 4 * subProblem.lowerBound / subProblem.path.Count;
        }

        private int GetParentIndex(int i)
        {
            return (i - 1) / 2;
        }

        private int GetLeftChildIndex(int i)
        {
            return 2 * i + 1;
        }

        private int GetRightChildIndex(int i)
        {
            return 2 * i + 2;
        }
    }



    public class SubProblem
    {
        public double lowerBound;
        public double[,] reducedMatrix;
        public List<int> path;
        private int size;

        public SubProblem(double[,] costMatrix, int size, double lowerBound, List<int> path)
        {
            this.path = path;
            this.lowerBound = lowerBound;
            this.size = size;
            this.reducedMatrix = this.ReduceCostMatrix(costMatrix);
        }


        public int GetCurrentCityIndex()
        {
            return this.path[this.path.Count - 1];
        }


        public bool HasMorePaths()
        {
            for (int col = 0; col < this.size; ++col)
            {
                if (this.reducedMatrix[this.GetCurrentCityIndex(), col] != double.PositiveInfinity)
                {
                    return true;
                }
            }
            return false;
        }


        public void BlockPaths(int i, int j)
        {
            this.lowerBound += this.reducedMatrix[i, j];
            for (int index = 0; index < this.size; ++index)
            {
                this.reducedMatrix[i, index] = double.PositiveInfinity;
                this.reducedMatrix[index, j] = double.PositiveInfinity;
            }
            this.reducedMatrix[j, i] = double.PositiveInfinity;
        }


        private double ReduceRow(double[,] matrix, int row)
        {
            // Find the lowest cost
            double lowestCost = double.PositiveInfinity;
            for (int col = 0; col < this.size; ++col)
            {
                if (matrix[row, col] < lowestCost)
                {
                    lowestCost = matrix[row, col];
                }
            }

            if (lowestCost == double.PositiveInfinity)
            {
                return 0;
            }

            for (int col = 0; col < this.size; ++col)
            {
                if (matrix[row, col] != double.PositiveInfinity)
                {
                    matrix[row, col] -= lowestCost;
                }
            }

            return lowestCost;
        }


        private double ReduceColumn(double[,] matrix, int col)
        {
            // Find the lowest cost
            double lowestCost = double.PositiveInfinity;
            for (int row = 0; row < this.size; ++row)
            {
                if (matrix[row, col] < lowestCost)
                {
                    lowestCost = matrix[row, col];
                }
            }

            if (lowestCost == double.PositiveInfinity)
            {
                return 0;
            }

            for (int row = 0; row < this.size; ++row)
            {
                if (matrix[row, col] != double.PositiveInfinity)
                {
                    matrix[row, col] -= lowestCost;
                }
            }

            return lowestCost;
        }


        public double[,] ReduceCostMatrix(double[,] matrix = null)
        {
            // Initially, we need to reduce the full cost matrix, but most of the time
            // this is just going to be re-reducing what's already there.
            if (matrix == null)
            {
                matrix = this.reducedMatrix;
            }
            
            double reductionCost = 0;
            for (int i = 0; i < this.size; ++i) {
                reductionCost += this.ReduceRow(matrix, i);
            }

            // Verify that every column has a 0 and reduce it if it doesn't
            for (int col = 0; col < this.size; ++col)
            {
                bool foundZero = false;
                for (int row = 0; row < this.size; ++row)
                {
                    if (matrix[row, col] == 0)
                    {
                        foundZero = true;
                    }
                }
                if (!foundZero)
                {
                    reductionCost += this.ReduceColumn(matrix, col);
                }
            }

            this.lowerBound += reductionCost;
            return matrix;
        }
    }
}
