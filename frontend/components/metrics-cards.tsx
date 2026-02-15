"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn, formatNumber, formatPercent } from "@/lib/utils";
import { AlertTriangle, CheckCircle, FlaskConical, Clock } from "lucide-react";
import type { AnalysisSummary } from "@/lib/api";

interface MetricsCardsProps {
  summary: AnalysisSummary | null;
  analysisTimeMs?: number;
  isLoading?: boolean;
}

export function MetricsCards({ summary, analysisTimeMs, isLoading }: MetricsCardsProps) {
  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {[1, 2, 3].map((i) => (
          <Card key={i} className="animate-pulse">
            <CardHeader className="pb-2">
              <div className="h-4 w-24 bg-muted rounded-lg" />
            </CardHeader>
            <CardContent>
              <div className="h-10 w-20 bg-muted rounded-lg" />
            </CardContent>
          </Card>
        ))}
      </div>
    );
  }

  if (!summary) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="bg-muted/50">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
              <FlaskConical className="h-4 w-4" />
              Comparisons
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-4xl font-bold text-muted-foreground">—</p>
          </CardContent>
        </Card>
        <Card className="bg-muted/50">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
              <AlertTriangle className="h-4 w-4" />
              Exceedances
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-4xl font-bold text-muted-foreground">—</p>
          </CardContent>
        </Card>
        <Card className="bg-muted/50">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
              <CheckCircle className="h-4 w-4" />
              Exceedance Rate
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-4xl font-bold text-muted-foreground">—</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  const hasExceedances = summary.exceedance_count > 0;

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {/* Total Comparisons */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
            <FlaskConical className="h-4 w-4" />
            Total Comparisons
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-4xl font-bold">{formatNumber(summary.total_comparisons)}</p>
          <p className="text-sm text-muted-foreground mt-1">
            {summary.total_samples} samples • {summary.total_parameters} parameters
          </p>
        </CardContent>
      </Card>

      {/* Exceedances */}
      <Card className={cn(hasExceedances && "metric-card-danger border-red-500")}>
        <CardHeader className="pb-2">
          <CardTitle
            className={cn(
              "text-sm font-medium flex items-center gap-2",
              hasExceedances ? "text-red-600 dark:text-red-400" : "text-muted-foreground"
            )}
          >
            <AlertTriangle className="h-4 w-4" />
            Exceedances
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p
            className={cn(
              "text-4xl font-bold",
              hasExceedances ? "text-red-600 dark:text-red-400" : ""
            )}
          >
            {formatNumber(summary.exceedance_count)}
          </p>
          <p className="text-sm text-muted-foreground mt-1">
            {summary.compliant_count} compliant • {summary.non_detect_count} non-detect
          </p>
        </CardContent>
      </Card>

      {/* Exceedance Rate */}
      <Card className={cn(hasExceedances && summary.exceedance_rate > 5 && "border-orange-400")}>
        <CardHeader className="pb-2">
          <CardTitle
            className={cn(
              "text-sm font-medium flex items-center gap-2",
              hasExceedances && summary.exceedance_rate > 5
                ? "text-orange-600 dark:text-orange-400"
                : "text-muted-foreground"
            )}
          >
            {hasExceedances ? (
              <AlertTriangle className="h-4 w-4" />
            ) : (
              <CheckCircle className="h-4 w-4" />
            )}
            Exceedance Rate
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p
            className={cn(
              "text-4xl font-bold",
              hasExceedances && summary.exceedance_rate > 5
                ? "text-orange-600 dark:text-orange-400"
                : hasExceedances
                ? "text-yellow-600"
                : "text-green-600"
            )}
          >
            {formatPercent(summary.exceedance_rate)}
          </p>
          {analysisTimeMs && (
            <p className="text-sm text-muted-foreground mt-1 flex items-center gap-1">
              <Clock className="h-3 w-3" />
              Analyzed in {(analysisTimeMs / 1000).toFixed(1)}s
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
