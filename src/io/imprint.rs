use crate::utils::Timer;
use log::{info, warn};

const CRATE_NAME: &str = env!("CARGO_PKG_NAME");
const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

const LOG_WIDTH: usize = 80;

pub fn write_header() {
    warn!("{: ^LOG_WIDTH$}", "-----------------");
    warn!("{: ^LOG_WIDTH$}", CRATE_NAME.to_uppercase());
    warn!("{: ^LOG_WIDTH$}", "-----------------");
    warn!("{: ^LOG_WIDTH$}", format!("version: {}", CRATE_VERSION));
    warn!("{: ^LOG_WIDTH$}", "");
    warn!("{: ^LOG_WIDTH$}", format!("{::^55}", ""));
    warn!(
        "{: ^80}",
        "::                   Roland Mitric                   ::"
    );
    warn!(
        "{: ^80}",
        "::  Institute of Physical and Theoretical Chemistry  ::"
    );
    warn!(
        "{: ^80}",
        "::              University of Wuerzburg              ::"
    );
    warn!(
        "{: ^80}",
        "::::::...................................................::::::"
    );
    warn!(
        "{: ^80}",
        ":: Contributors:                                             ::"
    );
    warn!(
        "{: ^80}",
        ":: --------                                                  ::"
    );
    warn!(
        "{: ^80}",
        ":: Richard Einsele       <richard.einsele@uni-wuerzburg.de>  ::"
    );
    warn!(
        "{: ^80}",
        ":: Joscha Hoche             <joscha.hoche@uni-wuerzburg.de>  ::"
    );
    warn!(
        "{: ^80}",
        ":: Xincheng Miao           <xincheng.miao@uni-wuerzburg.de>  ::"
    );
    warn!(
        "{: ^80}",
        ":: Luca Nils Philipp   <luca_nils.philipp@uni-wuerzburg.de>  ::"
    );
    warn!(
        "{: ^80}",
        ":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::"
    );
    warn!("{: ^80}", "");
}

pub fn write_footer(timer: Timer) {
    info!("{:^80}", "");
    warn!("{:-<80} ", "");
    warn!(
        "{:>68} {:>8.2} s",
        "total elapsed time:",
        timer.time.elapsed().as_secs_f32()
    );
    warn!("{: ^80}", "");
    warn!("{: ^80}", ":::::::::::::::::::::::::::::::::::::::");
    warn!(
        "{: ^80}",
        format!(
            "::    Thank you for using {}    ::",
            CRATE_NAME.to_uppercase()
        )
    );
    warn!("{: ^80}", ":::::::::::::::::::::::::::::::::::::::");
    warn!("{: ^80}", "");
}
