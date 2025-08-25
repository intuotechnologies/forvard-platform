import React, { useState } from "react";
import { motion } from "framer-motion";
import "../styles/documentation.css";

/* eslint-disable no-undef, no-sequences */

// Modulo per le referenze bibliografiche
const References = {
  "andersen2003": {
    authors: "Andersen, T. G., Bollerslev, T., Diebold, F. X., & Labys, P.",
    year: "2003",
    title: "Modeling and forecasting realized volatility",
    journal: "Econometrica", 
    volume: "71",
    pages: "579-625"
  },
  "andersen2012jump": {
    title: "Jump-robust volatility estimation using nearest neighbor truncation",
    journal: "Journal of Econometrics",
    volume: "169",
    number: "1",
    pages: "75-93",
    year: "2012",
    note: "Recent Advances in Panel Data, Nonlinear and Nonlinear and Nonparametric Models: A Festschrift in Honor of Peter C.B. Phillips",
    issn: "0304-4076",
    doi: "10.1016/j.jeconom.2012.01.011",
    authors: "Andersen, T. G.,  Dobrev, D. and Schaumburg, E.",
  },
  "andersen_bollerslev1998": {
    authors: "Andersen, T. G., Bollerslev, T.",
    year: "1998",
    title: "Answering the skeptics: Yes, standard volatility models do provide accurate forecasts",
    journal: "International Economic Review",
    volume: "39",
    pages: "885-905"
  },
  "bandi2008microstructure": {
    title: "Microstructure noise, realized variance, and optimal sampling",
    authors: "Bandi, F. M, and Russell, Jeffrey R",
    journal: "The Review of Economic Studies",
    volume: "75",
    number: "2",
    pages: "339-369",
    year: "2008",
    publisher: "Wiley-Blackwell"
  },
  "barndorff2002econometric": {
    title: "Econometric analysis of realized volatility and its use in estimating stochastic volatility models",
    authors: "Barndorff-Nielsen, O. E., and Shephard, N.",
    journal: "Journal of the Royal Statistical Society Series B: Statistical Methodology",
    volume: "64",
    number: "2",
    pages: "253--280",
    year: "2002",
    publisher: "Oxford University Press",
    doi: "10.1111/1467-9868.00336"
  },
  "barndorff2004covariation": {
    ISSN: "00129682, 14680262",
    authors: "Barndorff-Nielsen, O. E., and Shephard, N.",
    journal: "Econometrica",
    number: "3",
    pages: "885--925",
    publisher: "[Wiley, Econometric Society]",
    title: "Econometric Analysis of Realized Covariation: High Frequency Based Covariance, Regression, and Correlation in Financial Economics",
    urldate: "2025-05-21",
    volume: "72",
    year: "2004"
  },
  "barndorff2004measuring": {
    title: "Measuring the impact of jumps in multivariate price processes using bipower covariation",
    authors: "Barndorff-Nielsen, O. E., and Shephard, N.",
    year: "2004",
    institution: "Discussion paper, Nuffield College, Oxford University"
  },
  "barndorff2004power": {
    authors: "Barndorff-Nielsen, O. E., and Shephard, N.",
    title: "Power and Bipower Variation with Stochastic Volatility and Jumps",
    journal: "Journal of Financial Econometrics",
    volume: "2",
    number: "1",
    pages: "1-37",
    year: "2004",
    month: "01",
    issn: "1479-8409",
    doi: "10.1093/jjfinec/nbh001"
  },
  "barndorff2008designing": {
    authors: "Barndorff-Nielsen, O. E. and Hansen, P. R. and Lunde, A. and Shephard, N.",
    title: "Designing Realized Kernels to Measure the ex post Variation of Equity Prices in the Presence of Noise",
    journal: "Econometrica",
    volume: "76",
    number: "6",
    pages: "1481-1536",
    keywords: "Bipower variation, long-run variance estimator, market frictions, quadratic variation, realized variance",
    doi: "10.3982/ECTA6495",
    year: "2008"
  },
  "barndorff2009realized": {
    ISSN: "13684221, 1368423X",
    authors: "Barndorff-Nielsen, O. E., and Hansen, P. R. and Lunde, A. and Shephard, N.",
    journal: "The Econometrics Journal",
    number: "3",
    pages: "C1--C32",
    publisher: "[Royal Economic Society, Wiley]",
    title: "Realized kernels in practice: trades and quotes",
    volume: "12",
    doi: "10.1111/j.1368-423X.2008.00275.x",
    year: "2009"
  },
  "barndorff2010semivariance": {
    authors: "Barndorff‐Nielsen, O. E. and Kinnebrock, Silja and Shephard, N.",
    isbn: "9780199549498",
    title: "Measuring Downside Risk – Realized Semivariance",
    booktitle: "Volatility and Time Series Econometrics: Essays in Honor of Robert Engle",
    publisher: "Oxford University Press",
    year: "2010",
    month: "03",
    doi: "10.1093/acprof:oso/9780199549498.003.0007",
  },
  "barndorff2011multivariate": {
    title: "Multivariate realised kernels: Consistent positive semi-definite estimators of the covariation of equity prices with noise and non-synchronous trading",
    authors: "Barndorff-Nielsen, O. E. and Hansen, P. R. and Lunde, A. and Shephard, N.",
    journal: "Journal of Econometrics",
    volume: "162",
    number: "2",
    pages: "149--169",
    year: "2011",
    publisher: "Elsevier",
    doi: "10.1016/j.jeconom.2010.07.009"
  },
  "barndorff_shephard2002": {
    authors: "Barndorff-Nielsen, O. E., and Shephard, N.",
    year: "2002",
    title: "Econometric analysis of realized volatility and its use in estimating stochastic volatility models",
    journal: "Journal of the Royal Statistical Society",
    volume: "64",
    pages: "253-280"
  },
  "bollerslev2020realized": {
    title: "Realized semicovariances",
    authors: "Bollerslev, T. and Li, J. and Patton, A. J. and Quaedvlieg, R.",
    journal: "Econometrica",
    volume: "88",
    number: "4",
    pages: "1515--1551",
    year: "2020",
    publisher: "Wiley Online Library",
    doi: "10.3982/ECTA17056"
  },
  "brownlees_gallo2006": {
    authors: "Brownlees, C. T., and Gallo, G. M.",
    year: "2006",
    title: "Financial econometric analysis at ultra-high frequency: Data handling concerns",
    journal: "Computational Statistics & Data Analysis",
    volume: "51",
    pages: "2232-2245"
  },
  "christensen2007realized": {
    title: "Realized range-based estimation of integrated variance",
    authors: "Christensen, K. and Podolskij, M.",
    journal: "Journal of Econometrics",
    volume: "141",
    number: "2",
    pages: "323-349",
    year: "2007",
    issn: "0304-4076",
    doi: "10.1016/j.jeconom.2006.06.012"
  },
  "garman_klass1980": {
    authors: "Garman, M. B., and Klass, M. J.",
    year: "1980", 
    title: "On the estimation of security price volatilities from historical data",
    journal: "The Journal of Business",
    volume: "53",
    pages: "67-78"
  },
  "genccay2001introduction": {
    title: "An introduction to high-frequency finance",
    authors: "Gençay, Ramazan and Dacorogna, Michel and Muller, Ulrich A and Pictet, Olivier and Olsen, Richard",
    year: "2001",
    publisher: "Elsevier"
  },
  "hansen2006realized": {
    authors: "Hansen, P. R. and Lunde, A.",
    title: "Realized Variance and Market Microstructure Noise",
    journal: "Journal of Business \& Economic Statistics",
    volume: "24",
    number: "2",
    pages: "127-161",
    year: "2006",
    publisher: "ASA Website",
    doi: "10.1198/073500106000000071"
  },
  "martens2007measuring": { 
    title: "Measuring volatility with the realized range",
    authors: "Martens, M. and Van Dijk, D.",
    journal: "Journal of Econometrics",
    volume: "138",
    number: "1",
    pages: "181-207",
    year: "2007",
    publisher: "Elsevier",
    doi: "10.1016/j.jeconom.2006.05.019"
  },
  "parkinson1980": {
    authors: "Parkinson, M.",
    year: "1980",
    title: "The extreme value method for estimating the variance of the rate of return",
    journal: "The Journal of Business",
    volume: "53",
    pages: "61-65"
  },
  "zhang2005tale": {
    title: "A tale of two time scales: Determining integrated volatility with noisy high-frequency data",
    authors: "Zhang, L. and Mykland, P. A. and Aït-Sahalia, Y.",
    journal: "Journal of the American Statistical Association",
    volume: "100",
    number: "472",
    pages: "1394--1411",
    year: "2005",
    publisher: "Taylor \& Francis"
  },
  "zhang2011estimating": {
    title: "Estimating covariation: Epps effect, microstructure noise",
    authors: "Zhang, Lan",
    journal: "Journal of Econometrics",
    volume: "160",
    number: "1",
    pages: "33--47",
    year: "2011",
    publisher: "Elsevier",
    doi: "10.1016/j.jeconom.2010.03.012"
  },
  "zhou1996high": {
    authors: "Zhou, Bin",
    title: "High-Frequency Data and Volatility in Foreign-Exchange Rates",
    journal: "Journal of Business \& Economic Statistics",
    ISSN: "07350015",
    volume: "14",
    number: "1",
    pages: "45--52",
    year: "1996",
    publisher: "[American Statistical Association, Taylor & Francis, Ltd.]",
    doi: "10.1080/07350015.1996.10524628"
  }
};


{/*
// Componente per le formule matematiche
const MathFormula = ({ children, display = false }) => {
  if (display) {
    return (
      <div className="formula-description">
        <p><strong>{children}</strong></p>
      </div>
    );
  }
  return <code>{children}</code>;
};
*/}

// Componente per le formule matematiche
const MathFormula = ({ children, display = false }) => {
  if (display) {
    return (
      <div className="math-formula-display">
        {children}
      </div>
    );
  }
  return <span className="math-formula-inline">{children}</span>;
};

// Componente principale
const DocumentationPage = () => {
  const [activeSection, setActiveSection] = useState("the-library");

  const Reference = ({ id, children, type = "narrative" }) => {
  const ref = References[id];
  if (!ref) return <span style={{color: 'red', backgroundColor: '#fed7d7', padding: '2px 6px', borderRadius: '4px', fontSize: '0.8rem'}}>[REF NOT FOUND: {id}]</span>;
  
  const handleClick = (e) => {
    e.preventDefault();
    setActiveSection('references');
    setTimeout(() => {
      const element = document.getElementById(`ref-${id}`);
      if (element) {
        element.scrollIntoView({ 
          behavior: 'smooth', 
          block: 'center' 
        });
        element.style.backgroundColor = 'rgba(226, 255, 89, 0.3)';
        setTimeout(() => {
          element.style.backgroundColor = 'rgb(0, 78, 90)';
        }, 2000);
      }
    }, 100);
  };
  
  const formatAuthors = (authors) => {
    if (!authors) return 'Unknown';
    
    const authorList = authors.split(/ and |&/).map(author => author.trim().replace(/,$/, ''));
    
    if (authorList.length === 1) {
      const surname = authorList[0].split(',')[0].trim();
      return surname;
    } else if (authorList.length === 2) {
      const surname1 = authorList[0].split(',')[0].trim();
      const surname2 = authorList[1].split(',')[0].trim();
      return `${surname1} and ${surname2}`;
    } else {
      const surname1 = authorList[0].split(',')[0].trim();
      return `${surname1} et al.`;
    }
  };
  
  const authorDisplay = formatAuthors(ref.authors);
  
  // Formatta in base al tipo di citazione
  let citationText;
  if (type === "parenthetical") {
    citationText = `(${authorDisplay}, ${ref.year || 'N/A'})`;
  } else {
    citationText = `${authorDisplay} (${ref.year || 'N/A'})`;
  }
  
  return (
    <a 
      href={`#ref-${id}`}
      onClick={handleClick}
      title={`${ref.authors || 'Unknown'} (${ref.year || 'N/A'}). ${ref.title || 'No title'}`}
      style={{ cursor: 'pointer' }}
    >
      {children || citationText}
    </a>
  );
};
// Componente per citazioni multiple con link separati
const MultipleReferences = ({ ids, children }) => {
  const formatAuthors = (authors) => {
    if (!authors) return 'Unknown';
    
    const authorList = authors.split(/ and |&/).map(author => author.trim().replace(/,$/, ''));
    
    if (authorList.length === 1) {
      const surname = authorList[0].split(',')[0].trim();
      return surname;
    } else if (authorList.length === 2) {
      const surname1 = authorList[0].split(',')[0].trim();
      const surname2 = authorList[1].split(',')[0].trim();
      return `${surname1} and ${surname2}`;
    } else {
      const surname1 = authorList[0].split(',')[0].trim();
      return `${surname1} et al.`;
    }
  };
  
  const citations = ids.map((id, index) => {
    const ref = References[id];
    if (!ref) return <span key={id} style={{color: 'red'}}>[REF NOT FOUND: {id}]</span>;
    
    const handleClick = (e) => {
      e.preventDefault();
      setActiveSection('references');
      setTimeout(() => {
        const element = document.getElementById(`ref-${id}`);
        if (element) {
          element.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'center' 
          });
          element.style.backgroundColor = 'rgba(226, 255, 89, 0.3)';
          setTimeout(() => {
            element.style.backgroundColor = 'rgb(0, 78, 90)';
          }, 2000);
        }
      }, 100);
    };
    
    const authorDisplay = formatAuthors(ref.authors);
    const citationText = `${authorDisplay}, ${ref.year || 'N/A'}`;
    
    return (
      <span key={id}>
        <a 
          href={`#ref-${id}`}
          onClick={handleClick}
          title={`${ref.authors || 'Unknown'} (${ref.year || 'N/A'}). ${ref.title || 'No title'}`}
          style={{ cursor: 'pointer' }}
        >
          {citationText}
        </a>
        {index < ids.length - 1 && '; '}
      </span>
    );
  });
  
  return (
    <span>
      ({citations})
    </span>
  );
};

  const sections = [
    {
      id: "the-library",
      title: "The library",
      content: (
        <div>
          <h2>The library</h2>
          
          <p>
            The realized variance measures are computed using high-frequency data across different asset classes, including stocks, 
            exchange rates, and futures. Each of these asset classes has distinct characteristics, such as data type, 
            trading hours, and the treatment of outliers.
          </p>

          <h3>Stocks</h3>
          <p>
            For stocks, we use unadjusted prices of tick data with millisecond (ms) clock precision. The data for these 
            asset classes are recorded during regular market hours, which for U.S. equities are from 9:30:00.000 to 15:59:59.999 
            Eastern Time (ET). Additionally, since the U.S. market closes earlier at 13:00 on certain pre-holiday sessions 
            (such as the days preceding Christmas, Independence Day<sup>1</sup>, and the day following Thanksgiving), we consider data after 
            13:00 on these specific days as overnight activity. Before calculating the realized variance measures, outliers are detected 
            and cleaned from the data using the <Reference id="brownlees_gallo2006">Brownlees & Gallo (2006)</Reference> approach.
          </p>
          <p style={{fontSize: '0.9rem', color: '#666', lineHeight: '1.6', marginTop: '1rem', paddingLeft: '1rem', borderLeft: '3px solid #e2e8f0'}}>
            <sup>1</sup> In certain years, the U.S. stock market closes early (at 1 PM) on July 3rd ahead of Independence Day (July 4th). 
            However, when July 4th falls on a weekend (Saturday or Sunday) the holiday shifts either to the previous Friday or the 
            following Monday, respectively, and no early market closure occurs. Likewise, if July 4th is on a Monday, that day is 
            observed as a full holiday with no early closure on the preceding Friday.
          </p>

          <h3>Exchange rates</h3>
          <p>
            Exchange rates are gathered on the Forex market. For this market, the data used is bid-ask data with ms clock. 
            Unlike stocks, the Forex market operates continuously for 5 days a week, opening on Sunday around 5:00 PM ET 
            and closing on Friday around 5:00 PM ET. Due to the nature of the Forex market, we do not apply outlier detection to 
            the data, as large price movements are often part of the market's natural behavior and may not be considered outliers.
          </p>

          <h3>Futures</h3>
          <p>
            Similarly to Forex, futures data is bid-ask data with millisecond precision. While many futures contracts are available 
            for trading nearly 24 hours a day, they follow specific exchange hours rather than being truly continuous. Most major 
            futures contracts trade Sunday evening through Friday evening, with brief daily maintenance periods. Different contract 
            types (equity index futures, commodity futures, interest rate futures, etc.) follow their respective exchange schedules.
            For example, the CME Globex trading platform is open from Sunday evening to Friday afternoon. The trading hours are: 
            Sunday–Friday: 6 PM–5 PM ET, with a 60-minute break each day at 5 PM ET. As with the Forex market, no outlier detection 
            is performed for futures data, acknowledging the continuous and highly volatile nature of futures markets.
          </p>

          <p>
            The table below provides a summary of the key aspects of the data used for this library.
          </p>

          <div className="data-table-container">
            <table className="data-summary-table">
              <thead>
                <tr>
                  <th>Asset Class</th>
                  <th>Data Type</th>
                  <th>Trading Hours</th>
                  <th>Outliers detection</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Stocks</td>
                  <td>tick ms</td>
                  <td>9:30:00.000 - 15:59:59.999 (ET)</td>
                  <td>yes</td>
                </tr>
                <tr>
                  <td>ETFs</td>
                  <td>tick ms</td>
                  <td>9:30:00.000 - 15:59:59.999 (ET)</td>
                  <td>yes</td>
                </tr>
                <tr>
                  <td>Forex</td>
                  <td>bid ask ms</td>
                  <td>Sun 17:00 – Fri 17:00 ET</td>
                  <td>no</td>
                </tr>
                <tr>
                  <td>Futures</td>
                  <td>bid ask ms</td>
                  <td>Exchange-dependent (almost 24h)</td>
                  <td>no</td>
                </tr>
              </tbody>
            </table>
            <p className="table-caption">
              Summary of asset classes, data type, trading hours, and cleaning procedure. 
              The Kibot data, recorded with ms clock precision, contains overlapping timestamps and should be aggregated accordingly.
            </p>
          </div>


          <h2>Data</h2>
          <p>
            The data used for the evaluations are provided by <a href="https://www.kibot.com/" target="_blank" rel="noopener noreferrer">Kibot</a>. 
            For all asset classes we utilize data with millisecond clock, available since 2015.
           We download <strong>unadjusted</strong> data. Because the dataset contains millisecond precision, 
            there can be multiple entries with the same timestamp. In such cases, we aggregate the entries 
            while preserving the number of trades by adding an additional column to the original dataset.
          </p>
          <p>
            By default, for <strong>stocks and ETFs</strong>, the aggregation computes the 
            volume-weighted average price (VWAP) and sums both the total volume and the trade count. 
            For <strong>exchange rates and futures</strong>, in cases of repeated timestamps, 
            the aggregation uses the median for price, bid, and ask, and also accounts for 
            the number of trades.
          </p>
          <p>
            For the cleaning procedure and the estimation of realized variance measures, we restrict 
            the analysis to observations recorded during official market hours, 
            from 09:30:00.000 to 15:59:59.999. On certain pre-holiday sessions, such as the days preceding 
            Christmas and Independence Day, and the day following Thanksgiving, the U.S. market closes at 
            13:00; on these days, all activity after 13:00 is flagged as overnight trading.
          </p>
          <p>
            To ensure our dataset remains up to date, we perform <strong>periodic monthly updates</strong>.
          </p>
        </div>
      )
    },
    {
      id: "outlier-detection",
      title: "Outlier detection",
      content: (
        <div>
          <h2>Outlier detection</h2>
          
          <p>
            The procedure of <Reference id="brownlees_gallo2006" /> removes outlier price <MathFormula>pᵢ</MathFormula> if:
          </p>
          
          <MathFormula display>
            |pᵢ − p̄ᵢ(k)| ≥ 3sᵢ(k) + γ
          </MathFormula>
          
          <p>
            where <MathFormula>p̄ᵢ(k)</MathFormula> is the <MathFormula>δ</MathFormula>-trimmed sample mean of a neighborhood of k observations around <MathFormula>i</MathFormula>. 
            <MathFormula>sᵢ(k)</MathFormula> is the trimmed standard deviation of the same neighborhood. 
            <MathFormula>γ</MathFormula> is a granularity parameter to avoid zero variances from sequences of equal prices.
          </p>

          <h3>Step by step approach</h3>
          <ol>
            <li>
              <strong>Outlier detection rule.</strong> The outlier detection method takes four parameters: prices (an array of price values), 
              <MathFormula>k</MathFormula> (the window size), <MathFormula>δ</MathFormula> (the trimming proportion), and <MathFormula>γ</MathFormula> (a constant for outlier detection). For each price, the method 
              extracts the neighborhood of prices around the current index. It then calculates the trimmed mean and trimmed standard 
              deviation of this neighborhood. If the absolute difference between the current price and the trimmed mean is greater 
              than or equal to three times the trimmed standard deviation plus the constant gamma, the index is considered an outlier 
              and is added to the outliers list. The function ultimately returns the list of outliers and the DataFrame with the 
              outliers removed.
            </li>
            
            <li>
              <strong>Neighborhood definition.</strong> The procedure extracts the neighborhoods of prices around a given index <MathFormula>i</MathFormula> 
              in the prices array, while excluding the price at index <MathFormula>i</MathFormula> itself. The neighborhood is extracted using the following logic:
              <ul>
                <li>If <MathFormula>i</MathFormula> is less than <MathFormula>k/2</MathFormula>, it means that <MathFormula>i</MathFormula> is near the beginning of the array. In this case, the neighborhood is 
                taken from the start of the array up to <MathFormula>k + 1</MathFormula> elements.</li>
                <li>If <MathFormula>i</MathFormula> is greater than or equal to <MathFormula>n - k/2</MathFormula>, it means that <MathFormula>i</MathFormula> is near the end of the array. In this case, the 
                neighborhood is taken from the last <MathFormula>k + 1</MathFormula> elements of the array.</li>
                <li>Otherwise, the neighborhood is taken symmetrically around <MathFormula>i</MathFormula>, from <MathFormula>i - k/2</MathFormula> to <MathFormula>i + k/2 + 1</MathFormula>.</li>
              </ul>
              The function then removes the element at the center of the neighborhood (which corresponds to the original index <MathFormula>i</MathFormula>)
              ensuring that the price at index <MathFormula>i</MathFormula> is excluded from the neighborhood.
            </li>
            
            <li>
              <strong>Trimmed mean and standard deviation.</strong> To calculate the trimmed mean and trimmed standard deviation, 
              we begin by determining the number of values to be trimmed from each side of the sorted array. This is done by 
              calculating the trim count, which is the integer value of the length of the array multiplied by half of the trimming 
              proportion, <MathFormula>δ</MathFormula>. For instance, if <MathFormula>δ</MathFormula> is 0.2, the trim count will be 10% of the smallest values and 10% of the largest 
              values, resulting in 20% of the values being removed in total. Next, we sort the array in ascending order. Once sorted, 
              we remove the smallest and largest values based on the trim count. This results in a trimmed array that excludes the 
              specified proportion of extreme values, effectively reducing the influence of outliers. With the trimmed array, we then 
              compute the trimmed mean and standard deviation by taking the average and the standard deviation of the remaining values.
            </li>
            
            <li>
              <strong>Treating outliers.</strong> After identification of the outlier, the value is replaced by the mean between 
              the two previous and the two following prices.
            </li>
            
            <li>
              <strong>Parameter choice.</strong> <MathFormula>k</MathFormula> =120, <MathFormula>δ</MathFormula>= 0.1, <MathFormula>γ</MathFormula>=0.06
            </li>
          </ol>
        </div>
      )
    },
    {
      id: "sampling-method",
      title: "Sampling method",
      content: (
        <div>
          <h2>Sampling method</h2>
          
          <p>
            To address the challenge of irregular tick series in high-frequency financial data, it becomes necessary to sample 
            the tick series, transforming it into equally spaced series suitable for analysis and mitigating microstructure noise 
            effects <MultipleReferences ids={["genccay2001introduction", "hansen2006realized", "bandi2008microstructure"]} />.
            Among the various sampling methods proposed in the literature, the previous-tick method (using the last observed price prior to each sampling point) is widely 
            recognized for its efficacy and simplicity <Reference id="barndorff2009realized" type="parenthetical"/>.
          </p>
        </div>
      )
    },
    {
      id: "variance-measures",
      title: "Variance measures",
      content: (
        <div>
          <h2>Variance measures</h2>
          
          <p>
            We evaluate several variance estimators. In addition to the variance measures, the final processed file includes 
            additional information, such as the ticker, date, high price (H), low price (L), open price (O), close price (C)<sup>2</sup>, 
            and the total number of trades (N).
          </p>
          
          <p style={{fontSize: '0.9rem', color: '#666', lineHeight: '1.6', marginTop: '1rem', paddingLeft: '1rem', borderLeft: '3px solid #e2e8f0'}}>
            <sup>2</sup> We select H and L values from all tick data of the day, including odd lots. For the O and C prices, 
            we use the first and last available prices of the day's series excluding odd lots.
          </p>
          
          <div className="data-table-container">
            <table className="data-summary-table">
              <thead>
                <tr>
                  <th>Measure</th>
                  <th>Reference</th>
                  <th>Acronym</th>
                  <th>Subsampling</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Parkinson's Variance</td>
                  <td><Reference id="parkinson1980" /></td>
                  <td>pr</td>
                  <td>-</td>
                </tr>
                <tr>
                  <td>Garman-Klass's Variance</td>
                  <td><Reference id="garman_klass1980" /></td>
                  <td>gkr</td>
                  <td>-</td>
                </tr>
                <tr>
                  <td>Realized Range</td>
                  <td><Reference id="martens2007measuring" /><br/><Reference id="christensen2007realized" /></td>
                  <td>rr5</td>
                  <td>-</td>
                </tr>
                <tr>
                  <td>Realized Variance</td>
                  <td><Reference id="andersen_bollerslev1998" /><br/><Reference id="andersen2003" /></td>
                  <td>rv1, rv5</td>
                  <td>rv5_ss</td>
                </tr>
                <tr>
                  <td>Realized Quarticity</td>
                  <td><Reference id="barndorff2002econometric" /></td>
                  <td>rq1, rq5</td>
                  <td>rq5_ss</td>
                </tr>
                <tr>
                  <td>Bipower Variation</td>
                  <td><Reference id="barndorff2004power" /></td>
                  <td>bv1, bv5</td>
                  <td>bv5_ss</td>
                </tr>
                <tr>
                  <td>Realized Semivariance (pos e neg)</td>
                  <td><Reference id="barndorff2010semivariance" /></td>
                  <td>rsp1, rsp5, rsn1, rsn5</td>
                  <td>rsp5_ss, rsn5_ss</td>
                </tr>
                <tr>
                  <td>Median Realized Variance</td>
                  <td><Reference id="andersen2012jump" /></td>
                  <td>medRV1, medRV5</td>
                  <td>medRV5_ss</td>
                </tr>
                <tr>
                  <td>Minimum Realized Variance</td>
                  <td><Reference id="andersen2012jump" /></td>
                  <td>minRV1, minRV5</td>
                  <td>minRV5_ss</td>
                </tr>
                <tr>
                  <td>Realized Kernel</td>
                  <td><Reference id="barndorff2009realized" /></td>
                  <td>rk</td>
                  <td>-</td>
                </tr>
              </tbody>
            </table>
            <p className="table-caption">
              The number following the acronym indicates the time interval used to sample the series at 
              equally spaced points (e.g., rv1 refers to the realized variance sampling prices at 1-minute intervals). For each 
              subsampled realized measure, we evaluate it using 5 subsample sets, each shifted by 1 minute.
            </p>
          </div>

          <h3>Parkinson's Range (pr)</h3>
          <p>
            The Parkinson variance range <Reference id="parkinson1980" type="parenthetical"/> estimates volatility by utilizing the highest (H) and lowest (L) 
            prices observed within a given day. It is evaluated as:
          </p>
          <MathFormula display>
            <i>pr</i> = <span className="fraction">
              <span className="numerator">1</span>
              <span className="denominator">4 ln(2)</span>
            </span> ln<span className="thin-large-paren">(</span><span className="fraction">
              <span className="numerator">H</span>
              <span className="denominator">L</span>
            </span><span className="thin-large-paren">)</span><sup style={{fontSize: '0.6em', position: 'relative', top: '-0.8em'}}>2</sup>
          </MathFormula>

          <h3>Garman-Klass's Range (gkr)</h3>
          <p>
            The Garman-Klass variance range <Reference id="garman_klass1980" type="parenthetical" /> considers also the open (O) and close (C) prices 
            in addition to the high (H) and low (L) prices within a given day:
          </p>
          <MathFormula display>
          <i>gkr</i> = <span className="fraction">
            <span className="numerator">1</span>
            <span className="denominator">2</span>
          </span> ln<span className="thin-large-paren">(</span><span className="fraction">
            <span className="numerator">H</span>
            <span className="denominator">L</span>
          </span><span className="thin-large-paren">)</span><sup style={{fontSize: '0.6em', position: 'relative', top: '-0.8em'}}>2</sup> − (2ln(2) − 1) ln<span className="thin-large-paren">(</span><span className="fraction">
            <span className="numerator">C</span>
            <span className="denominator">O</span>
          </span><span className="thin-large-paren">)</span><sup style={{fontSize: '0.6em', position: 'relative', top: '-0.8em'}}>2</sup>
          </MathFormula>

          <h3>Realized Range (rr5)</h3>
          <p>
            The Realized Range introduced by <Reference id="martens2007measuring" /> and <Reference id="christensen2007realized" /> is a widely used 
            volatility estimator, especially for high-frequency trading data:
          </p>
          <MathFormula display>
          <i>rr5</i> = <span className="fraction">
            <span className="numerator">1</span>
            <span className="denominator">4 ln(2)</span>
          </span> <span className="summation">
            Σ
            <span className="sum-limits sum-upper">m</span>
            <span className="sum-limits sum-lower">i=1</span>
          </span> ln<span className="thin-large-paren">(</span><span className="fraction">
            <span className="numerator">Hᵢ</span>
            <span className="denominator">Lᵢ</span>
          </span><span className="thin-large-paren">)</span><sup style={{fontSize: '0.6em', position: 'relative', top: '-0.8em'}}>2</sup>
        </MathFormula>
          <p>
            <MathFormula>Hᵢ</MathFormula> and <MathFormula>Lᵢ</MathFormula> represent the highest and the lowest prices in the i-th interval. 
            <MathFormula>m</MathFormula> represents the number of intraday intervals.
          </p>
          <p><strong>Implementation choices:</strong> As sampling interval for the high and low prices we consider 5 minutes.</p>

          <h3>Realized Variance (rv)</h3>
          <p>
            The plain vanilla realized variance (rv) <MultipleReferences ids={["andersen_bollerslev1998", "andersen2003"]} /> is defined as:
          </p>
          <MathFormula display>
          <i>rv</i> = <span className="summation">
            Σ
            <span className="sum-limits sum-upper">m</span>
            <span className="sum-limits sum-lower">i=1</span>
          </span> rᵢ<sup style={{fontSize: '0.6em', position: 'relative', top: '-0.1em'}}>2</sup>
          </MathFormula>
          <p>
            where <MathFormula>rᵢ</MathFormula> denote the return of the <MathFormula>i</MathFormula>-th specific interval<sup>3</sup> with <MathFormula>m</MathFormula> representing the number of intraday intervals.
          </p>
          <p style={{fontSize: '0.9rem', color: '#666', lineHeight: '1.6', marginTop: '1rem', paddingLeft: '1rem', borderLeft: '3px solid #e2e8f0'}}>
            <sup>3</sup> <MathFormula>rᵢ = pᵢ − pᵢ₋₁</MathFormula> and <MathFormula>pᵢ = ln(Pᵢ)</MathFormula>, where <MathFormula>P</MathFormula> is the price.
          </p>
          <p><strong>Implementation choices:</strong> We provide two realized variance measures, calculated by evaluating returns based on the last price within 1-minute and 5-minute sampling intervals.</p>

          <h3>Realized Quarticity (rq)</h3>
          <p>
            Following <Reference id="barndorff2002econometric" />, the realized quarticity (rq) is defined as:
          </p>
          <MathFormula display>
          <i>rq</i> = <span className="fraction">
            <span className="numerator">m</span>
            <span className="denominator">3</span>
          </span> <span className="summation">
            Σ
            <span className="sum-limits sum-upper">m</span>
            <span className="sum-limits sum-lower">i=1</span>
          </span> rᵢ<sup style={{fontSize: '0.6em', position: 'relative', top: '-0.1em'}}>4</sup>
          </MathFormula>
          <p>
            where <MathFormula>rᵢ</MathFormula> denotes intraday returns over <MathFormula>m</MathFormula> equally spaced intervals within a trading day. 
            The factor <MathFormula>m/3</MathFormula> acts as a normalization constant that adjusts for the sampling frequency and ensures an unbiased estimator of the fourth moment of returns.
          </p>
          <p><strong>Implementation choices:</strong> We provide two realized quarticity measures, calculated by evaluating returns based on the last price within 1-minute and 5-minute sampling intervals.</p>

          <h3>Bipower Variation (bv)</h3>
          <p>
            The Bipower Variation <Reference id="barndorff2004power" type="parenthetical"/> is a robust volatility estimator that captures the 
            continuous component of price movements. It is computed as the sum of absolute returns, scaled by a constant factor. 
            The formula for the bipower variation over a day <MathFormula>t</MathFormula> is:
          </p>
          <MathFormula display>
          <i>bv</i> = <span className="fraction">
            <span className="numerator">π</span>
            <span className="denominator">2</span>
          </span> <span className="summation">
            Σ
            <span className="sum-limits sum-upper">m</span>
            <span className="sum-limits sum-lower">i=2</span>
          </span> |rᵢ| |rᵢ₋₁|
          </MathFormula>
          <p><strong>Implementation choices:</strong> We provide two bipower variation measures, calculated by evaluating returns based on the last price within 1-minute and 5-minute sampling intervals.</p>

          <h3>Realized Semivariance (rsp, rsn)</h3>
          <p>
            Realized semivariance <Reference id="barndorff2010semivariance" type="parenthetical"/> decomposes price volatility into its upside and downside 
            components, providing critical insight into the asymmetric nature of financial market fluctuations. The downside 
            (upside) realized semivariance <MathFormula>rs⁻</MathFormula> (<MathFormula>rs⁺</MathFormula>) captures the variance contribution from negative (positive) returns, as defined by:
          </p>
          <MathFormula display>
            <i>rs</i><sup style={{fontSize: '0.5em', position: 'relative', top: '-0.5em', left: '-0.1em'}}>−</sup> = <span className="summation">
              Σ
              <span className="sum-limits sum-upper">m</span>
              <span className="sum-limits sum-lower">i=1</span>
            </span> rᵢ<sup style={{fontSize: '0.6em', position: 'relative', top: '-0.1em'}}>2</sup> I<sub style={{fontSize: '0.7em', position: 'relative', bottom: '-0.3em'}}>[rᵢ &lt; 0]</sub>,&nbsp;&nbsp;&nbsp;<i> rs</i><sup style={{fontSize: '0.5em', position: 'relative', top: '-0.5em', left: '-0.1em'}}>+</sup> = <span className="summation">
              Σ
              <span className="sum-limits sum-upper">m</span>
              <span className="sum-limits sum-lower">i=1</span>
            </span> rᵢ<sup style={{fontSize: '0.6em', position: 'relative', top: '-0.1em'}}>2</sup> I<sub style={{fontSize: '0.7em', position: 'relative', bottom: '-0.3em'}}>[rᵢ &gt; 0]</sub>
          </MathFormula>
          <p>
            The indicator functions <MathFormula>I<sub>[rᵢ &lt; 0]</sub></MathFormula>, <MathFormula>I<sub>[rᵢ &gt; 0]</sub></MathFormula>
             ensure that only negative or positive returns, respectively, contribute 
            to each measure. These components sum to the total realized variance: <MathFormula>rv = rs⁺ + rs⁻</MathFormula>.
          </p>
          <p><strong>Implementation choices:</strong> We provide both the positive and negative components of realized semivariance measures, calculated by evaluating returns based on the last price within 1-minute and 5-minute sampling intervals.</p>

          <h3>Median Realized Variance (medRV)</h3>
          <p>
            Median realized variance <Reference id="andersen2012jump" type="parenthetical"/> provides a robust alternative to standard realized variance estimators 
            by effectively filtering out jumps in the price process. It uses the median of three consecutive squared returns to 
            minimize the impact of outliers and price discontinuities, making it particularly valuable for distinguishing between 
            continuous volatility components and jump variations. Formally, the median realized variance is defined as:
          </p>
          <MathFormula display>
          <i>medRV</i> = <span className="fraction">
            <span className="numerator">π</span>
            <span className="denominator">6−4√3+π</span>
          </span> · <span className="fraction">
            <span className="numerator">m</span>
            <span className="denominator">m−2</span>
          </span> <span className="summation">
            Σ
            <span className="sum-limits sum-upper">m−1</span>
            <span className="sum-limits sum-lower">i=2</span>
          </span> med(|rᵢ₋₁|, |rᵢ|, |rᵢ₊₁|)<sup style={{fontSize: '0.6em', position: 'relative', top: '-0.1em'}}>2</sup>
          </MathFormula>
          <p>
            where <MathFormula>med(·)</MathFormula> denotes the median operator applied to three consecutive absolute returns. The scaling factor <MathFormula>π/(6−4√3+π)</MathFormula> 
            ensures consistency with the quadratic variation process under continuous diffusion, while the term <MathFormula>m/(m−2)</MathFormula> provides 
            a small-sample adjustment.
          </p>
          <p><strong>Implementation choices:</strong> We evaluate returns based on the last price within 1-minute and 5-minute sampling intervals.</p>

          <h3>Minimum Realized Variance (minRV)</h3>
          <p>
            Minimum realized variance <Reference id="andersen2012jump" type="parenthetical" /> is another robust estimator of realized variance designed to mitigate 
            the impact of jumps in high-frequency financial data. It utilizes the minimum of two adjacent squared returns, effectively 
            filtering out large price movements that might represent jumps rather than continuous volatility. The estimator is formally defined as:
          </p>
          <MathFormula display>
          <i>minRV</i> = <span className="fraction">
            <span className="numerator">π</span>
            <span className="denominator">π−2</span>
          </span> ·<span className="fraction">
            <span className="numerator">m</span>
            <span className="denominator">m−1</span>
          </span> <span className="summation">
            Σ
            <span className="sum-limits sum-upper">m</span>
            <span className="sum-limits sum-lower">i=2</span>
          </span> min(|rᵢ₋₁|, |rᵢ|)<sup style={{fontSize: '0.6em', position: 'relative', top: '-0.1em'}}>2</sup>
          </MathFormula>
          <p>
            The scaling factor <MathFormula>π/(π−2)</MathFormula> ensures consistency with the integrated variance under continuous diffusion, and <MathFormula>m/(m−1)</MathFormula> 
            provides a finite-sample adjustment.
          </p>
          <p><strong>Implementation choices:</strong> We evaluate returns based on the last price within 1-minute and 5-minute sampling intervals.</p>

          <h3>Realized Kernel (rk)</h3>
          <p>
            The realized kernel variance estimator, introduced by <MultipleReferences ids={["barndorff2008designing", "barndorff2009realized"]} />, provides a consistent and 
            efficient way to estimate integrated variance in the presence of market microstructure noise. Given high-frequency returns <MathFormula>rᵢ</MathFormula>, 
            the realized kernel estimator takes the form:
          </p>
          <MathFormula display>
          <i>rk</i> = <span className="summation">
            Σ
            <span className="sum-limits sum-upper">H</span>
            <span className="sum-limits sum-lower">h=−H</span>
          </span> k<span className="thin-large-paren">(</span><span className="fraction">
            <span className="numerator">h</span>
            <span className="denominator">H+1</span>
          </span><span className="thin-large-paren">)</span> γₕ,&nbsp;&nbsp;&nbsp;γₕ = <span className="summation">
            Σ
            <span className="sum-limits sum-upper">n</span>
            <span className="sum-limits sum-lower">i=|h|+1</span>
          </span> rᵢ rᵢ₋|ₕ|
          </MathFormula>
          <p>
            where <MathFormula>k(·)</MathFormula> is a kernel function and <MathFormula>H</MathFormula> is the bandwidth parameter. 
            For symmetric <MathFormula>k(·)</MathFormula> functions satisfying <MathFormula>k(0)=1</MathFormula>, which is typically the case in practice, this expression can be equivalently written as:
          </p>
          <MathFormula display>
          <i>rk</i> = γ₀ + 2<span className="summation">
            Σ
            <span className="sum-limits sum-upper">H</span>
            <span className="sum-limits sum-lower">h=1</span>
          </span> k<span className="thin-large-paren">(</span><span className="fraction">
            <span className="numerator">h</span>
            <span className="denominator">H+1</span>
          </span><span className="thin-large-paren">)</span> γₕ
          </MathFormula>
          <p>
            where <MathFormula>γ₀ = Σᵢ₌₁ᵐ rᵢ²</MathFormula> is the standard realized variance. 
            The choice of kernel function <MathFormula>k(·)</MathFormula> is crucial for the estimator's properties. Since different kernel functions offer varying trade-offs between bias, variance, and robustness, selecting 
            an appropriate kernel is essential. Popular choices include Parzen, Tukey-Hanning, and cubic kernels. In our estimate, we employ the Parzen kernel due to its excellent balance between efficiency and robustness. 
            <Reference id="barndorff2009realized"/> show that the Parzen kernel fulfills all necessary conditions to guarantee the positivity 
            of the estimator and consistent estimation of integrated variance in the presence of microstructure noise.
            The Parzen kernel function is given by: 
          </p>
          <MathFormula display>
            <div style={{display: 'flex', alignItems: 'center'}}>
              <span>k(x) = </span>
              <span style={{fontSize: '3.5em', lineHeight: '0.7'}}>{'{'}</span>
              <div style={{marginLeft: '8px', fontSize: '0.85em', lineHeight: '1.2'}}>
                <div style={{marginBottom: '2px'}}>1 − 6x<sup style={{fontSize: '0.6em', position: 'relative', top: '-0.5em'}}>2</sup> + 6x<sup style={{fontSize: '0.6em', position: 'relative', top: '-0.5em'}}>3</sup><span style={{marginLeft: '15px', fontSize: '0.8em'}}>0 ≤ x ≤ <span className="fraction"><span className="numerator">1</span><span className="denominator">2</span></span></span></div>
                <div style={{marginBottom: '2px'}}>2(1 − x)<sup style={{fontSize: '0.6em', position: 'relative', top: '-0.5em'}}>3</sup><span style={{marginLeft: '30px', fontSize: '0.8em'}}><span className="fraction"><span className="numerator">1</span><span className="denominator">2</span></span> &lt; x ≤ 1</span></div>
                <div>0<span style={{marginLeft: '75px', fontSize: '0.8em'}}>x &gt; 1</span></div>
              </div>
            </div>
          </MathFormula>
          <p>
            This satisfies the smoothness conditions <MathFormula>k'(0) = k'(1) = 0</MathFormula> and is guaranteed to produce a non-negative estimate.
          </p>
          <p>
            We select the bandwidth as:
          </p>
          <MathFormula display>
          H* = c* ξ<sup style={{fontSize: '0.6em', position: 'relative', top: '-0.8em'}}>4/5</sup> n<sup style={{fontSize: '0.6em', position: 'relative', top: '-0.8em'}}>3/5</sup>
          </MathFormula>
          <p>with</p>
          <MathFormula display>
            c* = [<span className="fraction">
              <span className="numerator">k''(0)<sup style={{fontSize: '0.6em', position: 'relative', top: '-0.1em'}}>2</sup></span>
              <span className="denominator">k₀,₀</span>
            </span>]<sup style={{fontSize: '0.6em', position: 'relative', top: '-0.1em'}}>1/5</sup>&nbsp;&nbsp;&nbsp;and&nbsp;&nbsp;&nbsp;ξ<sup style={{fontSize: '0.6em', position: 'relative', top: '-0.1em'}}>2</sup> = <span className="fraction">
              <span className="numerator">ω<sup style={{fontSize: '0.6em', position: 'relative', top: '-0.1em'}}>2</sup></span>
              <span className="denominator">√(T ∫<sub style={{fontSize: '0.7em', position: 'relative', bottom: '-0.3em'}}>0</sub><sup style={{fontSize: '0.6em', position: 'relative', top: '-0.1em'}}>T</sup> σᵤ<sup style={{fontSize: '0.6em', position: 'relative', top: '-0.1em'}}>4</sup> du)</span>
            </span> = <span className="fraction">
              <span className="numerator">ω<sup style={{fontSize: '0.6em', position: 'relative', top: '-0.1em'}}>2</sup></span>
              <span className="denominator">√(T · IQ)</span>
            </span> ≈ <span className="fraction">
              <span className="numerator">ω<sup style={{fontSize: '0.6em', position: 'relative', top: '-0.1em'}}>2</sup></span>
              <span className="denominator">IV</span>
            </span>
            </MathFormula>
          <p>
            Where <MathFormula>c*</MathFormula> depends on <MathFormula>k''(0)²</MathFormula>, which is the squared second derivative of the kernel function evaluated at zero, and 
            <MathFormula>k₀,₀ = ||k||₂² = ∫ k(x)² dx</MathFormula> represents the squared L²-norm of the kernel function over its support. For the Parzen 
            kernel <MathFormula>c* = ((12)<sup style={{fontSize: '0.6em', position: 'relative', top: '-0.8em'}}>2</sup>/0.269)<sup style={{fontSize: '0.6em', position: 'relative', top: '-0.8em'}}>1/5</sup> = 3.5134</MathFormula>.
            The approximation <MathFormula>√(T × IQ) ≈ IV</MathFormula> is justified under conditions of moderate volatility variation. When volatility is 
            approximately constant, we have <MathFormula>IQ = ∫ σᵤ⁴ du ≈ (IV/T)²</MathFormula>, making <MathFormula>√(T · IQ) ≈ IV</MathFormula>. <Reference id="barndorff2009realized" /> 
            note this simplification is reasonable in practical applications and makes bandwidth selection more stable, as estimating 
            <MathFormula>IV</MathFormula> is significantly easier than estimating integrated quarticity.
            The parameter <MathFormula>ω²</MathFormula> is the noise variance, representing the variance of market microstructure noise that contaminates 
            observed prices. The bandwidth <MathFormula>H*</MathFormula> depends on this noise variance and the integrated quarticity <MathFormula>IQ = ∫ σᵤ⁴ du</MathFormula>, 
            while <MathFormula>IV = ∫ σᵤ² du</MathFormula> denotes the integrated variance.
            We estimate numerator and denominator of <MathFormula>ξ²</MathFormula> as follow.</p>

          <h4>Noise variance</h4>
          <p>
            According to <Reference id="zhang2005tale" /> and <Reference id="bandi2008microstructure" /> we estimate the noise variance as realized variance computed 
            using 2-minutes returns, <MathFormula>RV₂'</MathFormula>, divided by the number of non-zero returns, n:
          </p>
          <MathFormula display>
          ω̂<sup style={{fontSize: '0.6em', position: 'relative', top: '-0.8em'}}>2</sup> = <span className="fraction">
            <span className="numerator">RV<sub style={{fontSize: '0.7em', position: 'relative', bottom: '-0.3em'}}>2'</sub></span>
            <span className="denominator">2n</span>
          </span>
          </MathFormula>

          <h4>Integrated variance</h4>
          <p>
            According to <Reference id="barndorff2009realized" /> we estimate the integrated variance considering the realized variance 
            estimator based on 20 minutes returns by shifting the time of the first observation by one second. More specifically, 
            if <MathFormula>RV⁽ʲ⁾</MathFormula> represents the realized variance computed from the j-th subsample then:
          </p>
          <MathFormula display>
          ÎV = RV<sub>sparse</sub> = <span className="fraction">
            <span className="numerator">1</span>
            <span className="denominator">m</span>
          </span> · <span className="summation">
            Σ
            <span className="sum-limits sum-upper">m</span>
            <span className="sum-limits sum-lower">j=1</span>
          </span> RV<sup style={{fontSize: '0.6em', position: 'relative', top: '-0.8em'}}>(j)</sup>
          </MathFormula>
          <p>where <MathFormula>m</MathFormula> (=1200) denotes the number of subsamples.</p>

          <h4>End effect</h4>
          <p>
            Boundary treatment plays a crucial role in realized kernel estimation, as bias tends to accumulate at the edges of the 
            sampling period. <Reference id="barndorff2009realized" /> suggest using jittering techniques and endpoint corrections to address 
            boundary effects. These methods involve adjusting the first and last observations before sampling prices, using the 
            average of the two preceding and two succeeding prices to mitigate any edge-related bias.
          </p>

          <p><strong>Implementation choices:</strong> For the realized kernel we evaluate returns based on the last price within 1-second interval.</p>
        </div>
      )
    },
    {
      id: "subsampled-measures",
      title: "Subsampled measures",
      content: (
        <div>
          <h2>Subsampled measures</h2>
          
          <p>
            For some of the previously discussed measures (e.g., rv, rq, bv, rsp, rsn, medrv, minrv), we also provide the subsampled 
            measure <MultipleReferences ids={["zhou1996high", "zhang2005tale"]} />. The core idea behind this technique is to estimate the realized variance of 
            a time series using a sliding window that is subsampled multiple times within the window. This approach is particularly 
            useful in addressing challenges in high-frequency data analysis, such as microstructure noise. By averaging across 
            multiple subsamples, we obtain a more efficient estimator with lower variance, reducing the impact of noise and 
            mitigating potential bias.
          </p>
          
          <p>
            The subsample realized variance strikes a balance between using very high-frequency data, which contains more information 
            but also more noise, and lower-frequency data, which is less noisy but provides fewer data points. This method enhances 
            the quality of the realized variance estimator while addressing the trade-off between information content and noise.
          </p>
          
          <p>
            In financial econometrics, the typical approach to calculating the subsample realized variance follows these steps:
          </p>
          <ol>
            <li>Start with a fixed sampling frequency (e.g., every 5 minutes)</li>
            <li>Create multiple subsamples by shifting the sampling grid by a fraction of the frequency (e.g., shift by 1 minute).</li>
            <li>Calculate realized measure for each subsample.</li>
            <li>Average these estimates to compute the subsample realized measure.</li>
          </ol>
          
          <h3>Example</h3>
          <p>
            Consider the evaluation of the first estimator using a price series sampled every 5 minutes, starting at 9:30 AM. 
            The series contains price points at 9:30, 9:35, 9:40, ..., 4:00 PM.
          </p>
          <p>
            For the second estimator, instead of starting at 9:30 AM, we begin at 9:31 AM, and use the same 5-minute grid. This 
            means that the first return is calculated at 9:31 AM. The procedure is repeated for all subsequent subsamples.
          </p>
          <p>
            The final estimator is obtained by averaging the realized variance estimates from each of these subsamples.
          </p>
          
          <h3>Implementation choices</h3>
          <p>
            We evaluate all realized variance measures by considering five subsampling sets, each shifted by 1 minute. In each set, 
            the last price is selected as described in the Sampling Method section. We then calculate the realized variance for each 
            subsample and average the results to derive the final measure. 
            The subsampled estimators provided are: <MathFormula>rv5_ss, rq5_ss, bv5_ss, rsp5_ss, rsn5_ss, medRV5_ss, minRV5_ss</MathFormula>.
          </p>
        </div>
      )
    },
    {
      id: "covariance-measures",
      title: "Covariance measures",
      content: (
        <div>
          <h2>Covariance measures</h2>
          
          <p>
            A key challenge in estimating realized covariance from high-frequency financial data is the lack of synchronicity 
            between price observations across assets. Two common approaches to address this issue are: (i) constructing regularly 
            spaced time series by sampling the last observed price before each time point of a fixed time grid (known as the 
            "previous-tick" method) <Reference id="zhang2011estimating" type="parenthetical" />; and (ii) the refresh time scheme, which aligns observations based on 
            the first instance when all assets have updated prices <Reference id="barndorff2011multivariate" type="parenthetical" />.
          </p>
          
          <p>
            In this work we adopt the previous-tick method. Specifically, we sample the most recent transaction price available 
            before each time point of a predefined, equally spaced 1 minute time grid. This procedure yields a synchronized, 
            regularly spaced return series across all assets, allowing the use of standard realized covariance estimators.
          </p>
          
          <p>
            We introduce the notation used below. 
            We consider <MathFormula>N</MathFormula> assets observed at <MathFormula>m</MathFormula> equally-spaced intervals during a single trading day. Let <MathFormula><strong>r</strong>ₖ = [r₁,ₖ, r₂,ₖ, ..., r<sub>N,k</sub>]'</MathFormula> denote the  <MathFormula>N × 1</MathFormula> vector of returns during 
            the k-th intraday interval, where <MathFormula>k = 1, 2, ..., m.</MathFormula>
          </p>
          
          <p>
            Across all intervals, the collection of the collection of <MathFormula><strong>r</strong>ₖ</MathFormula> (<MathFormula>k = 1, 2, ..., m</MathFormula>) forms an <MathFormula>N × m</MathFormula> matrix of intraday returns. All 
            measures are constructed for a single trading day, with the time subscript suppressed for clarity. The table below summarizes the covariance measures calculated.
          </p>
          
          <div className="data-table-container">
            <table className="data-summary-table">
              <thead>
                <tr>
                  <th>Measure</th>
                  <th>Reference</th>
                  <th>Acronym</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Realized Covariance</td>
                  <td><Reference id="barndorff2004covariation" /></td>
                  <td>RCov</td>
                </tr>
                <tr>
                  <td>Bipower Covariation</td>
                  <td><Reference id="barndorff2004measuring" /></td>
                  <td>RBPCov</td>
                </tr>
                <tr>
                  <td>Realized Semicovariance<br/>(pos, neg, mixed)</td>
                  <td><Reference id="bollerslev2020realized" /></td>
                  <td>RSCov_P, RSCov_N,<br/>RSCov_Mp, RSCov_Mn</td>
                </tr>
              </tbody>
            </table>
            <p className="table-caption">
              Covariance measures implemented. For all estimates, we synchronize the time series and 
              sample them at 1-minute intervals to obtain equally spaced data.
            </p>
          </div>
          
          <p>
            We provide realized covariance matrices for groups of assets categorized by type (e.g., equities, exchange rates, 
            and futures).
          </p>

          <h3>Realized Covariance (RCov)</h3>
          <p>
            Following <Reference id="barndorff2004covariation" />, the realized covariance between two assets i and j over a single 
            trading day is defined as the sum of the products of their synchronized intraday returns:
          </p>
          <MathFormula display>
          RCov(ᵢ,ⱼ) = <span className="summation">
            Σ
            <span className="sum-limits sum-upper">m</span>
            <span className="sum-limits sum-lower">k=1</span>
          </span> rᵢ,ₖ · rⱼ,ₖ
          </MathFormula>
          <p>
            where <MathFormula>rᵢ,ₖ</MathFormula> and <MathFormula>rⱼ,ₖ</MathFormula> denote the returns of assets i and j, respectively, during the k-th intraday interval, 
            and <MathFormula>m</MathFormula> is the total number of intervals.
          </p>
          <p>
            For a portfolio of <MathFormula>N</MathFormula> assets, the realized covariance matrix aggregates all pairwise covariances into a single <MathFormula>N × N</MathFormula> 
            matrix, and is given by:
          </p>
          <MathFormula display>
          <strong>RCov</strong> = <span className="summation">
            Σ
            <span className="sum-limits sum-upper">m</span>
            <span className="sum-limits sum-lower">k=1</span>
          </span> <strong>r</strong>ₖ <strong>r</strong>ₖ'
          </MathFormula>

          <h3>Realized Bipower Covariation (RBPCov)</h3>
          <p>
            The realized bipower covariance is a robust estimator of integrated covariance that is less sensitive to jumps 
            compared to the standard realized covariance. Following <Reference id="barndorff2004measuring" />, the bipower covariance 
            between assets i and j is defined as:
          </p>
          <MathFormula display>
          RBPCV(i,j) = <span className="fraction">
            <span className="numerator">μ₁<sup style={{fontSize: '0.6em', position: 'relative', top: '-0.8em'}}>-2</sup></span>
            <span className="denominator">4</span>
          </span> · <span className="summation">
            Σ
            <span className="sum-limits sum-upper">m</span>
            <span className="sum-limits sum-lower">k=2</span>
          </span> (|rᵢ,ₖ₋₁ + rⱼ,ₖ₋₁| · |rᵢ,ₖ + rⱼ,ₖ| − |rᵢ,ₖ₋₁ − rⱼ,ₖ₋₁| · |rᵢ,ₖ − rⱼ,ₖ|)
          </MathFormula>
          <p>
            with <MathFormula>μ₁ = √(2/π) ≈ 0.7979</MathFormula> being the first moment of the absolute value of a standard normal random variable. 
            When <MathFormula>i = j</MathFormula>, this formulation simplifies to the standard realized bipower variation of asset <MathFormula>i</MathFormula>.
          </p>

          <h3>Realized Semicovariances (RSCov)</h3>
          <p>
            Realized semicovariances extend the traditional covariance framework by decomposing co-movements between assets based 
            on the signs of their returns, thereby uncovering asymmetric dependence structures that vary across market conditions.
          </p>
          <p>
            Following <Reference id="bollerslev2020realized" />, we decompose the realized covariance matrix into four components based on the 
            signs of the underlying high-frequency returns. We first define the signed return vectors as:
          </p>
          <MathFormula display>
          <strong>r</strong>ₖ⁺ = <strong>r</strong>ₖ ⊙ 𝟙[<strong>r</strong>ₖ {'>'} 𝟘]
          </MathFormula>
          <MathFormula display>
          <strong>r</strong>ₖ⁻ = <strong>r</strong>ₖ ⊙ 𝟙[<strong>r</strong>ₖ ≤ 𝟘]
          </MathFormula>
          <p>
            where <MathFormula>⊙</MathFormula> denotes element-wise multiplication and <MathFormula>𝟙[·]</MathFormula> is the indicator function applied element-wise. The four realized 
            semicovariance matrices are then defined as:
          </p>
          <MathFormula display>
          <strong>P</strong> = <strong>RSCov_P</strong> = <span className="summation">
            Σ
            <span className="sum-limits sum-upper">m</span>
            <span className="sum-limits sum-lower">k=1</span>
          </span> <strong>r</strong>ₖ⁺ (<strong>r</strong>ₖ⁺)' <span style={{fontSize: '0.8em', color: '#666'}}>(concordant positive)</span>
          </MathFormula>
          <MathFormula display>
          <strong>N</strong> = <strong>RSCov_N</strong> = <span className="summation">
            Σ
            <span className="sum-limits sum-upper">m</span>
            <span className="sum-limits sum-lower">k=1</span>
          </span> <strong>r</strong>ₖ⁻ (<strong>r</strong>ₖ⁻)' <span style={{fontSize: '0.8em', color: '#666'}}>(concordant negative)</span>
          </MathFormula>
          <MathFormula display>
          <strong>M</strong>⁺ = <strong>RSCov_Mp</strong> = <span className="summation">
            Σ
            <span className="sum-limits sum-upper">m</span>
            <span className="sum-limits sum-lower">k=1</span>
          </span> <strong>r</strong>ₖ⁺ (<strong>r</strong>ₖ⁻)' <span style={{fontSize: '0.8em', color: '#666'}}>(mixed/discordant positive-negative)</span>
          </MathFormula>
          <MathFormula display>
          <strong>M</strong>⁻ = <strong>RSCov_Mn</strong> = <span className="summation">
            Σ
            <span className="sum-limits sum-upper">m</span>
            <span className="sum-limits sum-lower">k=1</span>
          </span> <strong>r</strong>ₖ⁻ (<strong>r</strong>ₖ⁺)' <span style={{fontSize: '0.8em', color: '#666'}}>(mixed/discordant negative-positive)</span>
          </MathFormula>
          <p>
            The positive semicovariance <MathFormula><strong>P</strong></MathFormula> captures co-movement during periods when assets experience positive returns, 
            reflecting synchronized upward movements. The negative semicovariance matrix <MathFormula><strong>N</strong></MathFormula> measures co-movement during 
            joint downward movements, which is particularly relevant for risk management and portfolio diversification analysis. The 
            mixed semicovariance matrices <MathFormula><strong>M</strong>±</MathFormula> capture the co-movement when assets move in opposite directions.
          </p>
          <p>
            By construction, the standard realized covariance matrix can be decomposed as:
          </p>
          <MathFormula display>
          <strong>RCov</strong> = <strong>P</strong> + <strong>N</strong> + <strong>M</strong>⁺ + <strong>M</strong>⁻
          </MathFormula>
          <p>
            Note that while <MathFormula><strong>P</strong></MathFormula> and <MathFormula><strong>N</strong></MathFormula> are symmetric and positive semidefinite, the mixed semicovariance 
            matrices <MathFormula><strong>M</strong>⁺</MathFormula> and <MathFormula><strong>M</strong>⁻</MathFormula> are generally asymmetric, with zero diagonal elements by construction.
          </p>
        </div>
      )
    },
    {
      id: "references",
      title: "References",
      content: (
        <div>
          <h2>References</h2>
          <div style={{spaceY: '1.5rem'}}>
            {Object.entries(References).map(([key, ref]) => (
              <div key={key} id={`ref-${key}`} style={{marginBottom: '1.5rem', padding: '1rem', background: 'rgb(0, 78, 90)', borderRadius: '8px', borderLeft: '4px solid rgb(0, 78, 90)'}}>
                <div style={{lineHeight: '1.6', fontSize: '0.95rem'}}>
                  <strong>{ref.authors}</strong> ({ref.year}). {ref.title}. 
                  <em>{ref.journal}</em>
                  {ref.volume && `, ${ref.volume}`}
                  {ref.pages && `, ${ref.pages}`}.
                </div>
              </div>
            ))}
          </div>
        </div>
      )
    }
  ];

  const sectionVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { 
      opacity: 1, 
      y: 0, 
      transition: { duration: 0.6 } 
    },
  };

  return (
    <div className="documentation-page">
      <div className="doc-layout">
        <motion.nav 
          className="doc-sidebar"
          initial="hidden"
          animate="visible"
          variants={sectionVariants}
        >
          <ul>
            {sections.map((section) => (
              <li key={section.id}>
                <button
                  className={`nav-link ${activeSection === section.id ? 'active' : ''}`}
                  onClick={() => setActiveSection(section.id)}
                >
                  {section.title}
                </button>
              </li>
            ))}
          </ul>
        </motion.nav>

        <motion.main 
          className="doc-content"
          key={activeSection}
          initial="hidden"
          animate="visible"
          variants={sectionVariants}
        >
          {sections.find(section => section.id === activeSection)?.content}
        </motion.main>
      </div>
    </div>
  );
};

export default DocumentationPage;